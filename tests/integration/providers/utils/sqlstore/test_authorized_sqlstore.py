# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
from unittest.mock import patch

import pytest

from ogx.core.access_control.access_control import default_policy
from ogx.core.datatypes import User
from ogx.core.storage.datatypes import SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import (
    PostgresSqlStoreConfig,
    SqliteSqlStoreConfig,
    register_sqlstore_backends,
    sqlstore_impl,
)
from ogx_api.internal.sqlstore import ColumnType


def get_postgres_config():
    """Get PostgreSQL configuration if tests are enabled."""
    return PostgresSqlStoreConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        db=os.environ.get("POSTGRES_DB", "ogx"),
        user=os.environ.get("POSTGRES_USER", "ogx"),
        password=os.environ.get("POSTGRES_PASSWORD", "ogx"),
    )


def get_sqlite_config():
    """Get SQLite configuration with temporary file database."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()
    return SqliteSqlStoreConfig(db_path=temp_file.name)


# Backend configurations for parametrized tests
BACKEND_CONFIGS = [
    pytest.param(
        get_postgres_config,
        marks=pytest.mark.skipif(
            not os.environ.get("ENABLE_POSTGRES_TESTS"),
            reason="PostgreSQL tests require ENABLE_POSTGRES_TESTS environment variable",
        ),
        id="postgres",
    ),
    pytest.param(get_sqlite_config, id="sqlite"),
]


@pytest.fixture
def authorized_store(backend_config):
    """Set up authorized store with proper cleanup."""
    config_func = backend_config

    config = config_func()
    backend_name = f"sql_{type(config).__name__.lower()}"
    register_sqlstore_backends({backend_name: config})
    base_sqlstore = sqlstore_impl(SqlStoreReference(backend=backend_name, table_name="authorized_store"))
    authorized_store = AuthorizedSqlStore(base_sqlstore, default_policy())

    yield authorized_store

    if hasattr(config, "db_path"):
        try:
            os.unlink(config.db_path)
        except (OSError, FileNotFoundError):
            pass


async def create_test_table(authorized_store, table_name):
    """Create a test table with standard schema."""
    await authorized_store.create_table(
        table=table_name,
        schema={
            "id": ColumnType.STRING,
            "data": ColumnType.STRING,
        },
    )


async def cleanup_records(sql_store, table_name, record_ids):
    """Clean up test records."""
    for record_id in record_ids:
        try:
            await sql_store.delete(table_name, {"id": record_id})
        except Exception:
            pass


@pytest.mark.parametrize("backend_config", BACKEND_CONFIGS)
@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_authorized_store_attributes(mock_get_authenticated_user, authorized_store, request):
    """Test that JSON column comparisons work correctly for both PostgreSQL and SQLite"""
    backend_name = request.node.callspec.id

    # Create test table
    table_name = f"test_json_comparison_{backend_name}"
    await create_test_table(authorized_store, table_name)

    try:
        # Test with no authenticated user (should handle JSON null comparison)
        mock_get_authenticated_user.return_value = None

        # Insert some test data
        await authorized_store.insert(table_name, {"id": "1", "data": "public_data"})

        # Test fetching with no user - should not error on JSON comparison
        result = await authorized_store.fetch_all(table_name)
        assert len(result.data) == 1
        assert result.data[0]["id"] == "1"
        assert result.data[0]["access_attributes"] is None

        # Test with authenticated user
        test_user = User("test-user", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = test_user

        # Insert data with user attributes
        await authorized_store.insert(table_name, {"id": "2", "data": "admin_data"})

        # Fetch all - admin should see both
        result = await authorized_store.fetch_all(table_name)
        assert len(result.data) == 2

        # Test with non-admin user
        regular_user = User("regular-user", {"roles": ["user"]})
        mock_get_authenticated_user.return_value = regular_user

        # Should only see public record
        result = await authorized_store.fetch_all(table_name)
        assert len(result.data) == 1
        assert result.data[0]["id"] == "1"

        # Test the category missing branch: user with multiple attributes
        multi_user = User("multi-user", {"roles": ["admin"], "teams": ["dev"]})
        mock_get_authenticated_user.return_value = multi_user

        # Insert record with multi-user (has both roles and teams)
        await authorized_store.insert(table_name, {"id": "3", "data": "multi_user_data"})

        # Test different user types to create records with different attribute patterns
        # Record with only roles (teams category will be missing)
        roles_only_user = User("roles-user", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = roles_only_user
        await authorized_store.insert(table_name, {"id": "4", "data": "roles_only_data"})

        # Record with only teams (roles category will be missing)
        teams_only_user = User("teams-user", {"teams": ["dev"]})
        mock_get_authenticated_user.return_value = teams_only_user
        await authorized_store.insert(table_name, {"id": "5", "data": "teams_only_data"})

        # Record with different roles/teams (shouldn't match our test user)
        different_user = User("different-user", {"roles": ["user"], "teams": ["qa"]})
        mock_get_authenticated_user.return_value = different_user
        await authorized_store.insert(table_name, {"id": "6", "data": "different_user_data"})

        # Now test with the multi-user who has both roles=admin and teams=dev
        mock_get_authenticated_user.return_value = multi_user
        result = await authorized_store.fetch_all(table_name)

        # Should see:
        # - public record (1) - no access_attributes
        # - admin record (2) - user matches roles=admin, teams missing (allowed)
        # - multi_user record (3) - user matches both roles=admin and teams=dev
        # - roles_only record (4) - user matches roles=admin, teams missing (allowed)
        # - teams_only record (5) - user matches teams=dev, roles missing (allowed)
        # Should NOT see:
        # - different_user record (6) - user doesn't match roles=user or teams=qa
        expected_ids = {"1", "2", "3", "4", "5"}
        actual_ids = {record["id"] for record in result.data}
        assert actual_ids == expected_ids, f"Expected to see records {expected_ids} but got {actual_ids}"

        # Verify the category missing logic specifically
        # Records 4 and 5 test the "category missing" branch where one attribute category is missing
        category_test_ids = {record["id"] for record in result.data if record["id"] in ["4", "5"]}
        assert category_test_ids == {"4", "5"}, (
            f"Category missing logic failed: expected 4,5 but got {category_test_ids}"
        )

        # Test a user that has all roles and teams (should generate SQL)
        # owner_principal = ''
        # owner_principal = 'super-user'
        # ((JSON_EXTRACT(access_attributes, '$.roles') LIKE '%"admin"%') OR (JSON_EXTRACT(access_attributes, '$.roles') LIKE '%"user"%'))
        # ((JSON_EXTRACT(access_attributes, '$.teams') LIKE '%"dev"%') OR (JSON_EXTRACT(access_attributes, '$.teams') LIKE '%"qa"%'))
        super_user = User("super-user", {"roles": ["admin", "user"], "teams": ["dev", "qa"]})
        mock_get_authenticated_user.return_value = super_user
        result = await authorized_store.fetch_all(table_name)
        assert len(result.data) == 6

    finally:
        # Clean up records
        await cleanup_records(authorized_store.sql_store, table_name, ["1", "2", "3", "4", "5", "6"])


@pytest.mark.parametrize("backend_config", BACKEND_CONFIGS)
@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_user_ownership_policy(mock_get_authenticated_user, authorized_store, request):
    """Test that 'user is owner' policies work correctly with record ownership"""
    from ogx.core.access_control.datatypes import AccessRule, Action, Scope

    backend_name = request.node.callspec.id

    # Create test table
    table_name = f"test_ownership_{backend_name}"
    await create_test_table(authorized_store, table_name)

    try:
        # Test with first user who creates records
        user1 = User("user1", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = user1

        # Insert a record owned by user1
        await authorized_store.insert(table_name, {"id": "1", "data": "user1_data"})

        # Test with second user
        user2 = User("user2", {"roles": ["user"]})
        mock_get_authenticated_user.return_value = user2

        # Insert a record owned by user2
        await authorized_store.insert(table_name, {"id": "2", "data": "user2_data"})

        # Create a policy that only allows access when user is the owner
        owner_only_policy = [
            AccessRule(
                permit=Scope(actions=[Action.READ]),
                when=["user is owner"],
            ),
        ]

        # Create a new authorized store with the owner-only policy
        owner_only_store = AuthorizedSqlStore(authorized_store.sql_store, owner_only_policy)

        # Test user1 access - should only see their own record
        mock_get_authenticated_user.return_value = user1
        result = await owner_only_store.fetch_all(table_name)
        assert len(result.data) == 1, f"Expected user1 to see 1 record, got {len(result.data)}"
        assert result.data[0]["id"] == "1", f"Expected user1's record, got {result.data[0]['id']}"

        # Test user2 access - should only see their own record
        mock_get_authenticated_user.return_value = user2
        result = await owner_only_store.fetch_all(table_name)
        assert len(result.data) == 1, f"Expected user2 to see 1 record, got {len(result.data)}"
        assert result.data[0]["id"] == "2", f"Expected user2's record, got {result.data[0]['id']}"

        # Test with anonymous user - should see no records
        mock_get_authenticated_user.return_value = None
        result = await owner_only_store.fetch_all(table_name)
        assert len(result.data) == 0, f"Expected anonymous user to see 0 records, got {len(result.data)}"

    finally:
        # Clean up records
        await cleanup_records(authorized_store.sql_store, table_name, ["1", "2"])


@pytest.mark.parametrize("backend_config", BACKEND_CONFIGS)
@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_sqlrecord_created_with_no_owner(mock_get_authenticated_user, authorized_store, request):
    """Test that SqlRecord is created with no owner == None when owner_principal is empty/missing"""
    backend_name = request.node.callspec.id

    # Create test table
    table_name = f"test_sqlrecord_created_with_no_owner_{backend_name}"
    await create_test_table(authorized_store, table_name)

    try:
        # Test with no authenticated user (should handle JSON null comparison)
        mock_get_authenticated_user.return_value = None

        # Insert some test data
        await authorized_store.insert(table_name, {"id": "1", "data": "public_data"})

        # Test fetching with no user - should create SqlRecord with no owner
        with patch(
            "ogx.core.storage.sqlstore.authorized_sqlstore.is_action_allowed", return_value=True
        ) as mock_is_action_allowed:
            result = await authorized_store.fetch_all(table_name)
            mock_is_action_allowed.assert_called_once()
            args = mock_is_action_allowed.call_args
            assert args[0][2].type == f"sql_record::{table_name}"
            assert args[0][2].owner is None
        assert len(result.data) == 1

    finally:
        # Clean up records
        await cleanup_records(authorized_store.sql_store, table_name, ["1"])
