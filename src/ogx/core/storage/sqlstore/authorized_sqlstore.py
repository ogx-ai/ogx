# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

_VALID_JSON_PATH_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

from ogx.core.access_control.access_control import (
    ALLOWED_ATTRIBUTE_KEYS,
    AccessDeniedError,
    default_policy,
    is_action_allowed,
)
from ogx.core.access_control.conditions import ProtectedResource
from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.datatypes import User
from ogx.core.request_headers import get_authenticated_user
from ogx.core.storage.datatypes import StorageBackendType
from ogx.log import get_logger
from ogx_api import PaginatedResponse
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, SqlStore

logger = get_logger(name=__name__, category="providers::utils")

# Hardcoded copy of the default policy that our SQL filtering implements
# WARNING: If default_policy() changes, this constant must be updated accordingly
# or SQL filtering will fall back to conservative mode (safe but less performant)
#
# This policy represents: "Permit all actions when user is in owners list for ANY attribute category"
# The corresponding SQL logic is implemented in _build_default_policy_where_clause():
# - Public records (no access_attributes) are always accessible
# - Records with access_attributes require user to match ANY category that exists in the resource
# - Within each category, user needs ANY matching value (OR logic)
# - Between categories, user needs ANY category to match (OR logic)
SQL_OPTIMIZED_POLICY = [
    AccessRule(
        permit=Scope(actions=list(Action)),
        when=["user in owners " + name],
    )
    for name in ALLOWED_ATTRIBUTE_KEYS
] + [
    AccessRule(
        permit=Scope(actions=list(Action)),
        when=["user is owner"],
    ),
    AccessRule(
        permit=Scope(actions=list(Action)),
        when=["resource is unowned"],
    ),
]


def _enhance_item_with_access_control(item: Mapping[str, Any], current_user: User | None) -> Mapping[str, Any]:
    """Add access control attributes to a data item."""
    enhanced = dict(item)
    if current_user:
        enhanced["owner_principal"] = current_user.principal
        enhanced["access_attributes"] = current_user.attributes
    else:
        # IMPORTANT: Use empty string and null value (not None) to match public access filter
        # The public access filter in _get_public_access_conditions() expects:
        # - owner_principal = '' (empty string)
        # - access_attributes = null (JSON null, which serializes to the string 'null')
        # Setting them to None (SQL NULL) will cause rows to be filtered out on read.
        enhanced["owner_principal"] = ""
        enhanced["access_attributes"] = None  # Pydantic/JSON will serialize this as JSON null
    return enhanced


class SqlRecord(ProtectedResource):
    """A SQL record wrapped as a protected resource for access control checks."""

    def __init__(self, record_id: str, table_name: str, owner: User | None):
        self.type = f"sql_record::{table_name}"
        self.identifier = record_id
        self.owner = owner


class AuthorizedSqlStore:
    """
    Authorization layer for SqlStore that provides access control functionality.

    This class composes a base SqlStore and adds authorization methods that handle
    access control policies, user attribute capture, and SQL filtering optimization.
    """

    def __init__(self, sql_store: SqlStore, policy: list[AccessRule]):
        """
        Initialize the authorization layer.

        :param sql_store: Base SqlStore implementation to wrap
        :param policy: Access control policy to use for authorization
        """
        self.sql_store = sql_store
        self.policy = policy
        self._detect_database_type()
        self._validate_sql_optimized_policy()

    def _detect_database_type(self) -> None:
        """Detect the database type from the underlying SQL store."""
        if not hasattr(self.sql_store, "config"):
            raise ValueError("SqlStore must have a config attribute to be used with AuthorizedSqlStore")

        self.database_type = self.sql_store.config.type.value
        if self.database_type not in [StorageBackendType.SQL_POSTGRES.value, StorageBackendType.SQL_SQLITE.value]:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _validate_sql_optimized_policy(self) -> None:
        """Validate that SQL_OPTIMIZED_POLICY matches the actual default_policy().

        This ensures that if default_policy() changes, we detect the mismatch and
        can update our SQL filtering logic accordingly.
        """
        actual_default = default_policy()

        if SQL_OPTIMIZED_POLICY != actual_default:
            logger.warning(
                "SQL_OPTIMIZED_POLICY does not match default_policy(). SQL filtering will use conservative mode. Expected: , Got",
                sql_optimized_policy=SQL_OPTIMIZED_POLICY,
                actual_default=actual_default,
            )

    async def create_table(self, table: str, schema: Mapping[str, ColumnType | ColumnDefinition]) -> None:
        """Create a table with built-in access control support."""

        enhanced_schema = dict(schema)
        if "access_attributes" not in enhanced_schema:
            enhanced_schema["access_attributes"] = ColumnType.JSON
        if "owner_principal" not in enhanced_schema:
            enhanced_schema["owner_principal"] = ColumnType.STRING

        await self.sql_store.create_table(table, enhanced_schema)
        await self.sql_store.add_column_if_not_exists(table, "access_attributes", ColumnType.JSON)
        await self.sql_store.add_column_if_not_exists(table, "owner_principal", ColumnType.STRING)

    async def insert(self, table: str, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        """Insert a row or batch of rows with automatic access control attribute capture."""
        current_user = get_authenticated_user()
        enhanced_data: Mapping[str, Any] | Sequence[Mapping[str, Any]]
        if isinstance(data, Mapping):
            enhanced_data = _enhance_item_with_access_control(data, current_user)
        else:
            enhanced_data = [_enhance_item_with_access_control(item, current_user) for item in data]
        await self.sql_store.insert(table, enhanced_data)

    async def upsert(
        self,
        table: str,
        data: Mapping[str, Any],
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
    ) -> None:
        """Upsert a row with automatic access control attribute capture."""
        current_user = get_authenticated_user()
        enhanced_data = _enhance_item_with_access_control(data, current_user)
        await self.sql_store.upsert(
            table=table,
            data=enhanced_data,
            conflict_columns=conflict_columns,
            update_columns=update_columns,
        )

    async def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
        cursor: tuple[str, str] | None = None,
        action: Action = Action.READ,
    ) -> PaginatedResponse:
        """Fetch all rows with automatic access control filtering."""
        access_where, access_params = self._build_access_control_where_clause(self.policy)
        rows = await self.sql_store.fetch_all(
            table=table,
            where=where,
            where_sql=access_where,
            where_sql_params=access_params,
            limit=limit,
            order_by=order_by,
            cursor=cursor,
        )

        current_user = get_authenticated_user()
        filtered_rows = []

        for row in rows.data:
            stored_access_attrs = row.get("access_attributes")
            stored_owner_principal = row.get("owner_principal")

            record_id = row.get("id", "unknown")
            # Create owner as None if owner_principal is empty/missing, matching ResourceWithOwner behavior
            owner = (
                User(principal=stored_owner_principal, attributes=stored_access_attrs)
                if stored_owner_principal
                else None
            )
            sql_record = SqlRecord(str(record_id), table, owner)

            if is_action_allowed(self.policy, action, sql_record, current_user):
                filtered_rows.append(row)

        return PaginatedResponse(
            data=filtered_rows,
            has_more=rows.has_more,
        )

    async def fetch_one(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
        action: Action = Action.READ,
    ) -> dict[str, Any] | None:
        """Fetch one row with automatic access control checking."""
        results = await self.fetch_all(
            table=table,
            where=where,
            limit=1,
            order_by=order_by,
            action=action,
        )

        return results.data[0] if results.data else None

    async def update(self, table: str, data: Mapping[str, Any], where: Mapping[str, Any]) -> None:
        """Update rows with access control enforcement.

        Verifies the current user has UPDATE permission on existing rows before
        modifying them. Original ownership is preserved — updating a record does
        not transfer ownership to the caller.
        """
        current_user = get_authenticated_user()
        await self._check_access_for_rows(table, where, Action.UPDATE, current_user)

        enhanced_data = dict(data)
        enhanced_data.pop("owner_principal", None)
        enhanced_data.pop("access_attributes", None)
        if not enhanced_data:
            return

        if self._can_apply_sql_policy_filter_for_mutations(current_user):
            access_where, access_params = self._build_access_control_where_clause(self.policy)
            await self.sql_store.update(
                table,
                enhanced_data,
                where,
                where_sql=access_where,
                where_sql_params=access_params,
            )
            return

        await self.sql_store.update(table, enhanced_data, where)

    async def delete(self, table: str, where: Mapping[str, Any]) -> None:
        """Delete rows with access control enforcement.

        Verifies the current user has DELETE permission on existing rows before
        removing them. Raises AccessDeniedError if the user lacks permission.
        """
        current_user = get_authenticated_user()
        await self._check_access_for_rows(table, where, Action.DELETE, current_user)

        if self._can_apply_sql_policy_filter_for_mutations(current_user):
            access_where, access_params = self._build_access_control_where_clause(self.policy)
            await self.sql_store.delete(
                table,
                where,
                where_sql=access_where,
                where_sql_params=access_params,
            )
            return

        await self.sql_store.delete(table, where)

    async def _check_access_for_rows(
        self,
        table: str,
        where: Mapping[str, Any],
        action: Action,
        current_user: User | None,
    ) -> None:
        """Fetch rows matching `where` and verify the user has permission for `action` on each."""
        rows = await self.sql_store.fetch_all(table=table, where=where)
        for row in rows.data:
            record_id = row.get("id", "unknown")
            stored_owner_principal = row.get("owner_principal")
            stored_access_attrs = row.get("access_attributes")

            owner = (
                User(principal=stored_owner_principal, attributes=stored_access_attrs)
                if stored_owner_principal
                else None
            )
            sql_record = SqlRecord(str(record_id), table, owner)

            if not is_action_allowed(self.policy, action, sql_record, current_user):
                raise AccessDeniedError(action.value, sql_record, current_user)

    def _can_apply_sql_policy_filter_for_mutations(self, current_user: User | None) -> bool:
        """Return whether SQL-level policy filtering can be safely applied to update/delete."""
        return current_user is not None and (not self.policy or self.policy == SQL_OPTIMIZED_POLICY)

    def _build_access_control_where_clause(self, policy: list[AccessRule]) -> tuple[str, dict[str, Any]]:
        """Build SQL WHERE clause for access control filtering.

        Returns a tuple of (sql_clause, bind_params) using parameterized queries.
        Only applies SQL filtering for the default policy to ensure correctness.
        For custom policies, uses conservative filtering to avoid blocking legitimate access.
        """
        current_user = get_authenticated_user()

        if not policy or policy == SQL_OPTIMIZED_POLICY:
            return self._build_default_policy_where_clause(current_user)
        else:
            return self._build_conservative_where_clause()

    @staticmethod
    def _validate_json_path(path: str) -> None:
        if not _VALID_JSON_PATH_RE.match(path):
            raise ValueError(f"Invalid attribute key for JSON path: {path!r}")

    def _json_extract(self, column: str, path: str) -> str:
        """Extract JSON value (keeping JSON type).

        Args:
            column: The JSON column name
            path: The JSON path (e.g., 'roles', 'teams')

        Returns:
            SQL expression to extract JSON value
        """
        self._validate_json_path(path)
        if self.database_type == StorageBackendType.SQL_POSTGRES.value:
            return f"{column}->'{path}'"
        elif self.database_type == StorageBackendType.SQL_SQLITE.value:
            return f"JSON_EXTRACT({column}, '$.{path}')"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _json_extract_text(self, column: str, path: str) -> str:
        """Extract JSON value as text.

        Args:
            column: The JSON column name
            path: The JSON path (e.g., 'roles', 'teams')

        Returns:
            SQL expression to extract JSON value as text
        """
        self._validate_json_path(path)
        if self.database_type == StorageBackendType.SQL_POSTGRES.value:
            return f"{column}->>'{path}'"
        elif self.database_type == StorageBackendType.SQL_SQLITE.value:
            return f"JSON_EXTRACT({column}, '$.{path}')"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def _get_public_access_conditions(self) -> list[str]:
        """Get the SQL conditions for public access.

        Public records are those with:
        - owner_principal = '' (empty string)

        The policy "resource is unowned" only checks if owner_principal is empty,
        regardless of access_attributes.
        """
        return ["owner_principal = ''"]

    def _build_default_policy_where_clause(self, current_user: User | None) -> tuple[str, dict[str, Any]]:
        """Build SQL WHERE clause for the default policy.

        Returns a tuple of (sql_clause, bind_params) using parameterized queries.
        Default policy: permit all actions when user in owners [roles, teams, projects, namespaces]
        This means user must match ANY attribute category that exists in the resource (OR logic).
        """
        base_conditions = self._get_public_access_conditions()
        params: dict[str, Any] = {}

        if current_user:
            params["owner_principal_match"] = current_user.principal
            base_conditions.append("owner_principal = :owner_principal_match")

            if current_user.attributes:
                for attr_key, user_values in current_user.attributes.items():
                    if attr_key not in ALLOWED_ATTRIBUTE_KEYS:
                        logger.warning("Skipping unrecognized attribute key", attr_key=attr_key)
                        continue
                    if user_values:
                        value_conditions = []
                        for j, value in enumerate(user_values):
                            param_name = f"attr_{attr_key}_{j}"
                            json_text = self._json_extract_text("access_attributes", attr_key)
                            value_conditions.append(f"({json_text} LIKE :{param_name})")
                            params[param_name] = f'%"{value}"%'

                        if value_conditions:
                            base_conditions.append(f"({' OR '.join(value_conditions)})")

        return f"({' OR '.join(base_conditions)})", params

    def _build_conservative_where_clause(self) -> tuple[str, dict[str, Any]]:
        """Conservative SQL filtering for custom policies.

        Returns a tuple of (sql_clause, bind_params) using parameterized queries.
        Only filters records we're 100% certain would be denied by any reasonable policy.
        """
        current_user = get_authenticated_user()

        if not current_user:
            base_conditions = self._get_public_access_conditions()
            return f"({' OR '.join(base_conditions)})", {}

        return "1=1", {}
