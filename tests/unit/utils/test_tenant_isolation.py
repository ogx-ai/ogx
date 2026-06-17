# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from ogx.core.access_control.access_control import default_policy
from ogx.core.datatypes import TenancyMode, User
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.core.storage.sqlstore.sqlstore import SqliteSqlStoreConfig
from ogx_api import ConflictError
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType


def _make_store(
    tmp_dir: str,
    db_name: str,
    tenancy_mode: TenancyMode = TenancyMode.MULTI,
    default_tenant_id: str | None = None,
) -> AuthorizedSqlStore:
    base = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=f"{tmp_dir}/{db_name}"))
    return AuthorizedSqlStore(base, default_policy(), tenancy_mode, default_tenant_id)


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_tenant_a_cannot_read_tenant_b(mock_user):
    """Tenant A must not see tenant B's rows."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_read.db")
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        bob = User("bob", {"roles": ["admin"]}, tenant_id="tenant-b")

        mock_user.return_value = alice
        await store.insert("docs", {"id": "a1", "title": "Alice doc"})

        mock_user.return_value = bob
        await store.insert("docs", {"id": "b1", "title": "Bob doc"})

        mock_user.return_value = alice
        result = await store.fetch_all("docs")
        assert len(result.data) == 1
        assert result.data[0]["id"] == "a1"

        mock_user.return_value = bob
        result = await store.fetch_all("docs")
        assert len(result.data) == 1
        assert result.data[0]["id"] == "b1"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_tenant_a_cannot_update_tenant_b(mock_user):
    """Cross-tenant update must be denied."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_update.db")
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        bob = User("bob", {"roles": ["admin"]}, tenant_id="tenant-b")

        mock_user.return_value = alice
        await store.insert("docs", {"id": "a1", "title": "Original"})

        mock_user.return_value = bob
        await store.update("docs", {"title": "Hacked"}, where={"id": "a1"})

        # Verify nothing changed — bob's update should have matched zero rows
        mock_user.return_value = alice
        row = await store.fetch_one("docs", where={"id": "a1"})
        assert row is not None
        assert row["title"] == "Original"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_tenant_a_cannot_delete_tenant_b(mock_user):
    """Cross-tenant delete must be denied."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_delete.db")
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        bob = User("bob", {"roles": ["admin"]}, tenant_id="tenant-b")

        mock_user.return_value = alice
        await store.insert("docs", {"id": "a1", "title": "Keep me"})

        mock_user.return_value = bob
        await store.delete("docs", where={"id": "a1"})

        mock_user.return_value = alice
        row = await store.fetch_one("docs", where={"id": "a1"})
        assert row is not None


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_missing_tenant_returns_empty_in_multi_mode(mock_user):
    """In multi mode, a user with no tenant_id sees nothing (default deny)."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_deny.db")
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        mock_user.return_value = alice
        await store.insert("docs", {"id": "a1", "title": "Hello"})

        no_tenant = User("ghost", {"roles": ["admin"]})
        mock_user.return_value = no_tenant
        result = await store.fetch_all("docs")
        assert len(result.data) == 0


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_disabled_mode_ignores_tenant(mock_user):
    """In disabled mode, tenant_id on User is irrelevant — all rows visible."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_disabled.db", TenancyMode.DISABLED)
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        bob = User("bob", {"roles": ["admin"]}, tenant_id="tenant-b")

        mock_user.return_value = alice
        await store.insert("docs", {"id": "a1", "title": "Alice"})

        mock_user.return_value = bob
        await store.insert("docs", {"id": "b1", "title": "Bob"})

        # In disabled mode, both should be visible regardless of tenant
        mock_user.return_value = alice
        result = await store.fetch_all("docs")
        assert len(result.data) == 2


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_disabled_mode_preserves_application_tenant_id_column(mock_user):
    """In disabled mode, tenant_id is normal application data, not an isolation field."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_disabled_column.db", TenancyMode.DISABLED)
        await store.create_table(
            "docs",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "title": ColumnType.STRING,
                "tenant_id": ColumnType.STRING,
            },
        )

        mock_user.return_value = User("alice", {"roles": ["admin"]}, tenant_id="auth-tenant")
        await store.insert("docs", {"id": "doc1", "title": "Original", "tenant_id": "business-a"})
        raw = await store.sql_store.fetch_one("docs", where={"id": "doc1"})
        assert raw is not None
        assert raw["tenant_id"] == "business-a"

        await store.update("docs", {"tenant_id": "business-b"}, where={"id": "doc1"})
        raw = await store.sql_store.fetch_one("docs", where={"id": "doc1"})
        assert raw is not None
        assert raw["tenant_id"] == "business-b"

        await store.upsert(
            "docs",
            {"id": "doc1", "title": "Upserted", "tenant_id": "business-c"},
            conflict_columns=["id"],
        )
        raw = await store.sql_store.fetch_one("docs", where={"id": "doc1"})
        assert raw is not None
        assert raw["tenant_id"] == "business-c"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_single_mode_stamps_default_tenant(mock_user):
    """In single mode, all rows get the configured default tenant."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_single.db", TenancyMode.SINGLE)
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        user = User("alice", {"roles": ["admin"]}, tenant_id="default-tenant")
        mock_user.return_value = user
        await store.insert("docs", {"id": "a1", "title": "Doc"})

        # Read via raw store to check tenant_id was stamped
        raw = await store.sql_store.fetch_all("docs", where={"id": "a1"})
        assert raw.data[0]["tenant_id"] == "default-tenant"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_single_mode_requestless_write_uses_default_tenant(mock_user):
    """Requestless setup writes in single mode should land in the configured tenant."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_single_default.db", TenancyMode.SINGLE, "default-tenant")
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        mock_user.return_value = None
        await store.insert("docs", {"id": "startup", "title": "Startup doc"})
        result = await store.fetch_all("docs")
        assert len(result.data) == 1

        mock_user.return_value = User("system", None, tenant_id="default-tenant")
        result = await store.fetch_all("docs")
        assert len(result.data) == 1
        assert result.data[0]["id"] == "startup"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_single_mode_backfills_existing_disabled_mode_rows(mock_user):
    """Enabling single tenancy should keep legacy disabled-mode rows visible."""
    with TemporaryDirectory() as tmp:
        base = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=f"{tmp}/tenant_migration.db"))
        disabled_store = AuthorizedSqlStore(base, default_policy(), TenancyMode.DISABLED)
        await disabled_store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        mock_user.return_value = User("alice", {"roles": ["admin"]})
        await disabled_store.insert("docs", {"id": "legacy", "title": "Legacy doc"})

        single_store = AuthorizedSqlStore(base, default_policy(), TenancyMode.SINGLE, "default-tenant")
        await single_store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        mock_user.return_value = User("alice", {"roles": ["admin"]}, tenant_id="default-tenant")
        result = await single_store.fetch_all("docs")
        assert len(result.data) == 1
        assert result.data[0]["id"] == "legacy"
        assert result.data[0]["tenant_id"] == "default-tenant"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_insert_strips_client_supplied_tenant_id(mock_user):
    """Client-supplied tenant_id in the data payload must be stripped."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_strip.db")
        await store.create_table("docs", {"id": ColumnType.STRING, "title": ColumnType.STRING})

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        mock_user.return_value = alice
        await store.insert("docs", {"id": "a1", "title": "Doc", "tenant_id": "evil-tenant"})

        raw = await store.sql_store.fetch_all("docs", where={"id": "a1"})
        assert raw.data[0]["tenant_id"] == "tenant-a"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_upsert_does_not_cross_tenant_boundary(mock_user):
    """Upsert with conflicting ID across tenants must surface a conflict."""
    with TemporaryDirectory() as tmp:
        store = _make_store(tmp, "tenant_upsert.db")
        await store.create_table(
            "docs",
            {"id": ColumnDefinition(type=ColumnType.STRING, primary_key=True), "title": ColumnType.STRING},
        )

        alice = User("alice", {"roles": ["admin"]}, tenant_id="tenant-a")
        bob = User("bob", {"roles": ["admin"]}, tenant_id="tenant-b")

        mock_user.return_value = alice
        await store.insert("docs", {"id": "shared-id", "title": "Alice original"})

        # Bob tries to upsert with the same ID. The conflict must not update
        # Alice's tenant partition or silently drop Bob's write.
        mock_user.return_value = bob
        with pytest.raises(ConflictError, match="Failed to upsert row"):
            await store.upsert("docs", {"id": "shared-id", "title": "Bob doc"}, conflict_columns=["id"])

        mock_user.return_value = alice
        result = await store.fetch_all("docs")
        assert len(result.data) == 1
        assert result.data[0]["title"] == "Alice original"


def test_user_positional_constructor_backward_compatibility():
    """User('alice', attrs) positional call must still work."""
    u = User("alice", {"roles": ["admin"]})
    assert u.principal == "alice"
    assert u.attributes == {"roles": ["admin"]}
    assert u.tenant_id is None


def test_user_with_tenant_id():
    """User with explicit tenant_id via keyword argument."""
    u = User("alice", {"roles": ["admin"]}, tenant_id="t1")
    assert u.principal == "alice"
    assert u.tenant_id == "t1"
    assert u.attributes == {"roles": ["admin"]}
