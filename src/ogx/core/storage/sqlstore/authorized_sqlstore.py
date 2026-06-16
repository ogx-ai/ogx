# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
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
from ogx.core.datatypes import TenancyConfig, TenancyMode, User
from ogx.core.request_headers import get_authenticated_user
from ogx.core.storage.datatypes import SqlStoreReference, StorageBackendType
from ogx.core.storage.sqlstore.sqlstore import _sqlstore_impl
from ogx.log import get_logger
from ogx_api import ConflictError, PaginatedResponse
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, DeleteOperation, SqlStore

logger = get_logger(name=__name__, category="providers::utils")

_default_tenancy_config: TenancyConfig = TenancyConfig()


def set_default_tenancy_config(config: TenancyConfig) -> None:
    """Set the process-wide tenancy config. Called once during stack initialization."""
    global _default_tenancy_config
    _default_tenancy_config = config


def get_default_tenancy_config() -> TenancyConfig:
    """Return the process-wide tenancy config set during stack initialization."""
    return _default_tenancy_config


def set_default_tenancy_mode(mode: TenancyMode) -> None:
    """Set the process-wide tenancy mode. Called once during stack initialization."""
    global _default_tenancy_config
    _default_tenancy_config = TenancyConfig.model_construct(mode=mode, default_tenant_id=None)


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


def _enhance_item_with_access_control(
    item: Mapping[str, Any],
    current_user: User | None,
    tenancy_mode: TenancyMode = TenancyMode.DISABLED,
    default_tenant_id: str | None = None,
) -> Mapping[str, Any]:
    """Add access control and tenant attributes to a data item."""
    enhanced = dict(item)
    # Never trust client-supplied access control fields.
    enhanced.pop("owner_principal", None)
    enhanced.pop("access_attributes", None)
    if tenancy_mode != TenancyMode.DISABLED:
        enhanced.pop("tenant_id", None)
    if current_user:
        enhanced["owner_principal"] = current_user.principal
        enhanced["access_attributes"] = current_user.attributes
        if tenancy_mode != TenancyMode.DISABLED:
            enhanced["tenant_id"] = current_user.tenant_id or default_tenant_id or ""
    else:
        enhanced["owner_principal"] = ""
        enhanced["access_attributes"] = None
        if tenancy_mode != TenancyMode.DISABLED:
            enhanced["tenant_id"] = default_tenant_id or ""
    return enhanced


class SqlRecord(ProtectedResource):
    """A SQL record wrapped as a protected resource for access control checks."""

    def __init__(self, record_id: str, table_name: str, owner: User | None):
        self.type = f"sql_record::{table_name}"
        self.identifier = record_id
        self.owner = owner


async def authorized_sqlstore(
    reference: SqlStoreReference, policy: list[AccessRule], tenancy_mode: TenancyMode | None = None
) -> "AuthorizedSqlStore":
    """Create an AuthorizedSqlStore from a store reference and access policy.

    This is the only supported way to obtain a SQL store for API use.
    When tenancy_mode is None, uses the process-wide default set during initialization.
    """
    mode = tenancy_mode if tenancy_mode is not None else _default_tenancy_config.mode
    default_tenant_id = _default_tenancy_config.default_tenant_id if tenancy_mode is None else None
    return AuthorizedSqlStore(await _sqlstore_impl(reference), policy, mode, default_tenant_id)


class AuthorizedSqlStore:
    """
    Authorization layer for SqlStore that provides access control functionality.

    This class composes a base SqlStore and adds authorization methods that handle
    access control policies, user attribute capture, and SQL filtering optimization.
    """

    def __init__(
        self,
        sql_store: SqlStore,
        policy: list[AccessRule],
        tenancy_mode: TenancyMode = TenancyMode.DISABLED,
        default_tenant_id: str | None = None,
    ):
        """
        Initialize the authorization layer.

        :param sql_store: Base SqlStore implementation to wrap
        :param policy: Access control policy to use for authorization
        :param tenancy_mode: Tenancy isolation mode
        :param default_tenant_id: Tenant ID for requestless writes in single-tenant mode
        """
        self.sql_store = sql_store
        self.policy = policy
        self.tenancy_mode = tenancy_mode
        self.default_tenant_id = default_tenant_id
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
        if self.tenancy_mode != TenancyMode.DISABLED and "tenant_id" not in enhanced_schema:
            enhanced_schema["tenant_id"] = ColumnType.STRING

        await self.sql_store.create_table(table, enhanced_schema)
        await self.sql_store.add_column_if_not_exists(table, "access_attributes", ColumnType.JSON)
        await self.sql_store.add_column_if_not_exists(table, "owner_principal", ColumnType.STRING)
        if self.tenancy_mode != TenancyMode.DISABLED:
            await self.sql_store.add_column_if_not_exists(table, "tenant_id", ColumnType.STRING)
            if self.tenancy_mode == TenancyMode.SINGLE and self.default_tenant_id:
                await self.sql_store.update(
                    table,
                    {"tenant_id": self.default_tenant_id},
                    where={"tenant_id": None},
                )

    async def add_column_if_not_exists(
        self,
        table: str,
        column_name: str,
        column_type: ColumnType,
        nullable: bool = True,
    ) -> None:
        """Expose schema migration helper from the wrapped SQL store."""
        await self.sql_store.add_column_if_not_exists(table, column_name, column_type, nullable)

    async def check_access_for_rows(
        self,
        table: str,
        where: Mapping[str, Any],
        action: Action,
    ) -> None:
        """Validate authorization for matching rows without mutating data."""
        current_user = get_authenticated_user()
        await self._check_access_for_rows(table, where, action, current_user)

    def _build_tenant_filter(self, current_user: User | None) -> tuple[str, dict[str, Any]]:
        """Non-bypassable tenant partition filter. Applied before ABAC."""
        if self.tenancy_mode == TenancyMode.DISABLED:
            return "1=1", {}
        if not current_user or not current_user.tenant_id:
            if self.tenancy_mode == TenancyMode.SINGLE and self.default_tenant_id:
                return "tenant_id = :_tenant_id_filter", {"_tenant_id_filter": self.default_tenant_id}
            return "1=0", {}
        return "tenant_id = :_tenant_id_filter", {"_tenant_id_filter": current_user.tenant_id}

    def _tenant_id_for_current_context(self, current_user: User | None) -> str | None:
        if self.tenancy_mode == TenancyMode.DISABLED:
            return None
        if current_user and current_user.tenant_id:
            return current_user.tenant_id
        if self.tenancy_mode == TenancyMode.SINGLE:
            return self.default_tenant_id
        return None

    def _user_for_policy(self, current_user: User | None) -> User | None:
        if self.tenancy_mode != TenancyMode.DISABLED or current_user is None:
            return current_user
        return User(principal=current_user.principal, attributes=current_user.attributes)

    async def _check_tenant_conflict_for_upsert(
        self,
        table: str,
        conflict_where: Mapping[str, Any],
        current_user: User | None,
    ) -> None:
        if self.tenancy_mode == TenancyMode.DISABLED or not conflict_where:
            return

        current_tenant_id = self._tenant_id_for_current_context(current_user)
        rows = await self.sql_store.fetch_all(table=table, where=conflict_where)
        for row in rows.data:
            if row.get("tenant_id") != current_tenant_id:
                raise ConflictError(
                    f"Failed to upsert row in {table}: conflict columns match an existing row in another tenant"
                )

    def _combine_where_clauses(self, *clauses: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """Combine multiple SQL WHERE clauses with AND."""
        parts = []
        params: dict[str, Any] = {}
        for sql, sql_params in clauses:
            if sql and sql != "1=1":
                parts.append(f"({sql})")
                params.update(sql_params)
        if not parts:
            return "1=1", {}
        return " AND ".join(parts), params

    async def insert(self, table: str, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        """Insert a row or batch of rows with automatic access control attribute capture."""
        current_user = get_authenticated_user()
        enhanced_data: Mapping[str, Any] | Sequence[Mapping[str, Any]]
        if isinstance(data, Mapping):
            enhanced_data = _enhance_item_with_access_control(
                data,
                current_user,
                self.tenancy_mode,
                self.default_tenant_id,
            )
        else:
            enhanced_data = [
                _enhance_item_with_access_control(item, current_user, self.tenancy_mode, self.default_tenant_id)
                for item in data
            ]
        await self.sql_store.insert(table, enhanced_data)

    async def upsert(
        self,
        table: str,
        data: Mapping[str, Any],
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
    ) -> None:
        """Upsert a row with access control enforcement.

        Verifies the current user has UPDATE permission on any existing row
        matching the conflict columns before upserting. Original ownership is
        preserved on conflict - upserting a record does not transfer ownership
        to the caller.
        """
        current_user = get_authenticated_user()

        conflict_where = {col: data[col] for col in conflict_columns if col in data}
        if conflict_where:
            await self._check_access_for_rows(table, conflict_where, Action.UPDATE, current_user)
            await self._check_tenant_conflict_for_upsert(table, conflict_where, current_user)

        enhanced_data = _enhance_item_with_access_control(
            data,
            current_user,
            self.tenancy_mode,
            self.default_tenant_id,
        )

        frozen_fields = {"owner_principal", "access_attributes"}
        if self.tenancy_mode != TenancyMode.DISABLED:
            frozen_fields.add("tenant_id")
        if update_columns is not None:
            update_columns = [c for c in update_columns if c not in frozen_fields]
        else:
            update_columns = [c for c in enhanced_data.keys() if c not in conflict_columns and c not in frozen_fields]

        tenant_update_where, tenant_update_params = self._build_tenant_filter(current_user)

        await self.sql_store.upsert(
            table=table,
            data=enhanced_data,
            conflict_columns=conflict_columns,
            update_columns=update_columns,
            update_where_sql=tenant_update_where if tenant_update_where != "1=1" else None,
            update_where_sql_params=tenant_update_params if tenant_update_params else None,
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
        current_user = get_authenticated_user()
        access_where, access_params = self._build_access_control_where_clause(self.policy)
        tenant_where, tenant_params = self._build_tenant_filter(current_user)
        combined_where, combined_params = self._combine_where_clauses(
            (access_where, access_params),
            (tenant_where, tenant_params),
        )
        rows = await self.sql_store.fetch_all(
            table=table,
            where=where,
            where_sql=combined_where,
            where_sql_params=combined_params,
            limit=limit,
            order_by=order_by,
            cursor=cursor,
        )

        filtered_rows = []
        policy_user = self._user_for_policy(current_user)

        for row in rows.data:
            stored_access_attrs = row.get("access_attributes")
            stored_owner_principal = row.get("owner_principal")

            record_id = row.get("id", "unknown")
            owner = (
                User(
                    principal=stored_owner_principal,
                    attributes=stored_access_attrs,
                    tenant_id=row.get("tenant_id") if self.tenancy_mode != TenancyMode.DISABLED else None,
                )
                if stored_owner_principal
                else None
            )
            sql_record = SqlRecord(str(record_id), table, owner)

            if is_action_allowed(self.policy, action, sql_record, policy_user):
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
        if self.tenancy_mode != TenancyMode.DISABLED:
            enhanced_data.pop("tenant_id", None)
        if not enhanced_data:
            return

        tenant_where, tenant_params = self._build_tenant_filter(current_user)

        if self._can_apply_sql_policy_filter_for_mutations(current_user):
            access_where, access_params = self._build_access_control_where_clause(self.policy)
            combined_where, combined_params = self._combine_where_clauses(
                (access_where, access_params),
                (tenant_where, tenant_params),
            )
            await self.sql_store.update(
                table,
                enhanced_data,
                where,
                where_sql=combined_where,
                where_sql_params=combined_params,
            )
            return

        if tenant_where != "1=1":
            await self.sql_store.update(
                table,
                enhanced_data,
                where,
                where_sql=tenant_where,
                where_sql_params=tenant_params,
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

        tenant_where, tenant_params = self._build_tenant_filter(current_user)

        if self._can_apply_sql_policy_filter_for_mutations(current_user):
            access_where, access_params = self._build_access_control_where_clause(self.policy)
            combined_where, combined_params = self._combine_where_clauses(
                (access_where, access_params),
                (tenant_where, tenant_params),
            )
            await self.sql_store.delete(
                table,
                where,
                where_sql=combined_where,
                where_sql_params=combined_params,
            )
            return

        if tenant_where != "1=1":
            await self.sql_store.delete(
                table,
                where,
                where_sql=tenant_where,
                where_sql_params=tenant_params,
            )
            return

        await self.sql_store.delete(table, where)

    async def delete_many(self, operations: Sequence[DeleteOperation]) -> None:
        """Delete multiple row sets atomically with access control enforcement."""
        if not operations:
            return

        current_user = get_authenticated_user()
        for operation in operations:
            await self._check_access_for_rows(operation.table, operation.where, Action.DELETE, current_user)

        if self._can_apply_sql_policy_filter_for_mutations(current_user):
            access_where, access_params = self._build_access_control_where_clause(self.policy)
            filtered_operations = [
                DeleteOperation(
                    table=operation.table,
                    where=operation.where,
                    where_sql=(
                        access_where if operation.where_sql is None else f"({operation.where_sql}) AND ({access_where})"
                    ),
                    where_sql_params={**(operation.where_sql_params or {}), **access_params},
                )
                for operation in operations
            ]
            await self.sql_store.delete_many(filtered_operations)
            return

        await self.sql_store.delete_many(operations)

    async def _check_access_for_rows(
        self,
        table: str,
        where: Mapping[str, Any],
        action: Action,
        current_user: User | None,
    ) -> None:
        """Fetch rows matching `where` and verify the user has permission for `action` on each."""
        tenant_where, tenant_params = self._build_tenant_filter(current_user)
        rows = await self.sql_store.fetch_all(
            table=table,
            where=where,
            where_sql=tenant_where if tenant_where != "1=1" else None,
            where_sql_params=tenant_params if tenant_params else None,
        )
        policy_user = self._user_for_policy(current_user)
        for row in rows.data:
            record_id = row.get("id", "unknown")
            stored_owner_principal = row.get("owner_principal")
            stored_access_attrs = row.get("access_attributes")

            owner = (
                User(
                    principal=stored_owner_principal,
                    attributes=stored_access_attrs,
                    tenant_id=row.get("tenant_id") if self.tenancy_mode != TenancyMode.DISABLED else None,
                )
                if stored_owner_principal
                else None
            )
            sql_record = SqlRecord(str(record_id), table, owner)

            if not is_action_allowed(self.policy, action, sql_record, policy_user):
                raise AccessDeniedError(action.value, sql_record, policy_user)

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

    def _json_array_contains_value(self, column: str, path: str, param_name: str, value: str) -> tuple[str, Any]:
        """Generate SQL condition and bind param for checking if a JSON array contains an exact value.

        Args:
            column: The JSON column name
            path: The JSON path to the array (e.g., 'roles', 'teams')
            param_name: Name for the bind parameter
            value: The exact value to check for in the array

        Returns:
            A tuple of (sql_condition, param_value)
        """
        self._validate_json_path(path)
        if self.database_type == StorageBackendType.SQL_POSTGRES.value:
            sql = f"CAST({column}->'{path}' AS jsonb) @> CAST(:{param_name} AS jsonb)"
            return sql, json.dumps([value])
        elif self.database_type == StorageBackendType.SQL_SQLITE.value:
            sql = f"EXISTS (SELECT 1 FROM json_each(json_extract({column}, '$.{path}')) WHERE value = :{param_name})"
            return sql, value
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
                            condition, param_value = self._json_array_contains_value(
                                "access_attributes", attr_key, param_name, value
                            )
                            value_conditions.append(f"({condition})")
                            params[param_name] = param_value

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
