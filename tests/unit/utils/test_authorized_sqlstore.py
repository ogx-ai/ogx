# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from ogx.core.access_control.access_control import AccessDeniedError, default_policy, is_action_allowed
from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import User
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore, SqlRecord
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.core.storage.sqlstore.sqlstore import SqliteSqlStoreConfig
from ogx_api.internal.sqlstore import ColumnType


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_authorized_fetch_with_where_sql_access_control(mock_get_authenticated_user):
    """Test that fetch_all works correctly with where_sql for access control"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_access_control.db"
        base_sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        # Create table with access control
        await sqlstore.create_table(
            table="documents",
            schema={
                "id": ColumnType.INTEGER,
                "title": ColumnType.STRING,
                "content": ColumnType.TEXT,
            },
        )

        admin_user = User("admin-user", {"roles": ["admin"], "teams": ["engineering"]})
        regular_user = User("regular-user", {"roles": ["user"], "teams": ["marketing"]})

        # Set user attributes for creating documents
        mock_get_authenticated_user.return_value = admin_user

        # Insert documents with access attributes
        await sqlstore.insert("documents", {"id": 1, "title": "Admin Document", "content": "This is admin content"})

        # Change user attributes
        mock_get_authenticated_user.return_value = regular_user

        await sqlstore.insert("documents", {"id": 2, "title": "User Document", "content": "Public user content"})

        # Test that access control works with where parameter
        mock_get_authenticated_user.return_value = admin_user

        # Admin should see both documents
        result = await sqlstore.fetch_all("documents", where={"id": 1})
        assert len(result.data) == 1
        assert result.data[0]["title"] == "Admin Document"

        # User should only see their document
        mock_get_authenticated_user.return_value = regular_user

        result = await sqlstore.fetch_all("documents", where={"id": 1})
        assert len(result.data) == 0

        result = await sqlstore.fetch_all("documents", where={"id": 2})
        assert len(result.data) == 1
        assert result.data[0]["title"] == "User Document"

        row = await sqlstore.fetch_one("documents", where={"id": 1})
        assert row is None

        row = await sqlstore.fetch_one("documents", where={"id": 2})
        assert row is not None
        assert row["title"] == "User Document"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_sql_policy_consistency(mock_get_authenticated_user):
    """Test that SQL WHERE clause logic exactly matches is_action_allowed policy logic"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_consistency.db"
        base_sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="resources",
            schema={
                "id": ColumnType.STRING,
                "name": ColumnType.STRING,
            },
        )

        # Test scenarios with different access control patterns
        test_scenarios = [
            # Scenario 1: Public record (no access control - represents None user insert)
            {"id": "1", "name": "public", "owner_principal": "", "access_attributes": None},
            # Scenario 2: Record with roles requirement
            {"id": "2", "name": "admin-only", "owner_principal": "owner1", "access_attributes": {"roles": ["admin"]}},
            # Scenario 3: Record with multiple attribute categories
            {
                "id": "3",
                "name": "admin-ml-team",
                "owner_principal": "owner2",
                "access_attributes": {"roles": ["admin"], "teams": ["ml-team"]},
            },
            # Scenario 4: Record with teams only (missing roles category)
            {
                "id": "4",
                "name": "ml-team-only",
                "owner_principal": "owner3",
                "access_attributes": {"teams": ["ml-team"]},
            },
            # Scenario 5: Record with roles and projects
            {
                "id": "5",
                "name": "admin-project-x",
                "owner_principal": "owner4",
                "access_attributes": {"roles": ["admin"], "projects": ["project-x"]},
            },
        ]

        for scenario in test_scenarios:
            await base_sqlstore.insert("resources", scenario)

        # Test with different user configurations
        user_scenarios = [
            # User 1: No attributes (should only see public records)
            {"principal": "user1", "attributes": None},
            # User 2: Empty attributes (should only see public records)
            {"principal": "user2", "attributes": {}},
            # User 3: Admin role only
            {"principal": "user3", "attributes": {"roles": ["admin"]}},
            # User 4: ML team only
            {"principal": "user4", "attributes": {"teams": ["ml-team"]}},
            # User 5: Admin + ML team
            {"principal": "user5", "attributes": {"roles": ["admin"], "teams": ["ml-team"]}},
            # User 6: Admin + Project X
            {"principal": "user6", "attributes": {"roles": ["admin"], "projects": ["project-x"]}},
            # User 7: Different role (should only see public)
            {"principal": "user7", "attributes": {"roles": ["viewer"]}},
        ]

        policy = default_policy()

        for user_data in user_scenarios:
            user = User(principal=user_data["principal"], attributes=user_data["attributes"])
            mock_get_authenticated_user.return_value = user

            sql_results = await sqlstore.fetch_all("resources")
            sql_ids = {row["id"] for row in sql_results.data}
            policy_ids = set()
            for scenario in test_scenarios:
                # Create owner matching what was stored (None for public records)
                owner = (
                    User(principal=scenario["owner_principal"], attributes=scenario["access_attributes"])
                    if scenario["owner_principal"]
                    else None
                )
                sql_record = SqlRecord(
                    record_id=scenario["id"],
                    table_name="resources",
                    owner=owner,
                )

                if is_action_allowed(policy, Action.READ, sql_record, user):
                    policy_ids.add(scenario["id"])
            assert sql_ids == policy_ids, (
                f"Consistency failure for user {user.principal} with attributes {user.attributes}:\n"
                f"SQL returned: {sorted(sql_ids)}\n"
                f"Policy allows: {sorted(policy_ids)}\n"
                f"Difference: SQL only: {sql_ids - policy_ids}, Policy only: {policy_ids - sql_ids}"
            )


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_authorized_store_user_attribute_capture(mock_get_authenticated_user):
    """Test that user attributes are properly captured during insert"""
    with TemporaryDirectory() as tmp_dir:
        db_name = "test_attributes.db"
        base_sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        authorized_store = AuthorizedSqlStore(base_sqlstore, default_policy())

        await authorized_store.create_table(
            table="user_data",
            schema={
                "id": ColumnType.STRING,
                "content": ColumnType.STRING,
            },
        )

        mock_get_authenticated_user.return_value = User(
            "user-with-attrs", {"roles": ["editor"], "teams": ["content"], "projects": ["blog"]}
        )

        await authorized_store.insert("user_data", {"id": "item1", "content": "User content"})

        mock_get_authenticated_user.return_value = User("user-no-attrs", None)

        await authorized_store.insert("user_data", {"id": "item2", "content": "Public content"})

        mock_get_authenticated_user.return_value = None

        await authorized_store.insert("user_data", {"id": "item3", "content": "Anonymous content"})
        result = await base_sqlstore.fetch_all("user_data", order_by=[("id", "asc")])
        assert len(result.data) == 3

        # First item should have full attributes
        assert result.data[0]["id"] == "item1"
        assert result.data[0]["access_attributes"] == {"roles": ["editor"], "teams": ["content"], "projects": ["blog"]}

        # Second item should have null attributes (user with no attributes)
        assert result.data[1]["id"] == "item2"
        assert result.data[1]["access_attributes"] is None

        # Third item should have null attributes (no authenticated user)
        assert result.data[2]["id"] == "item3"
        assert result.data[2]["access_attributes"] is None


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_update_enforces_access_control(mock_get_authenticated_user):
    """Test that update() raises AccessDeniedError when user lacks permission."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_update_acl.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"], "teams": ["eng"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "title": "Original"})

        # A different user with non-overlapping attributes should be denied
        other_user = User("bob", {"roles": ["viewer"], "teams": ["marketing"]})
        mock_get_authenticated_user.return_value = other_user

        with pytest.raises(AccessDeniedError):
            await sqlstore.update("docs", {"title": "Hacked"}, where={"id": "doc1"})

        # Verify the row was not modified
        mock_get_authenticated_user.return_value = owner
        row = await sqlstore.fetch_one("docs", where={"id": "doc1"})
        assert row is not None
        assert row["title"] == "Original"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_update_preserves_ownership(mock_get_authenticated_user):
    """Test that update() does not transfer ownership to the calling user."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_update_ownership.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "title": "Original"})

        # A user with matching attributes can update, but ownership stays with alice
        teammate = User("carol", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = teammate
        await sqlstore.update("docs", {"title": "Updated"}, where={"id": "doc1"})

        # Read from the raw store to inspect ownership columns directly
        raw = await base_sqlstore.fetch_all("docs", where={"id": "doc1"})
        assert len(raw.data) == 1
        assert raw.data[0]["title"] == "Updated"
        assert raw.data[0]["owner_principal"] == "alice"
        assert raw.data[0]["access_attributes"] == {"roles": ["admin"]}


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_delete_enforces_access_control(mock_get_authenticated_user):
    """Test that delete() raises AccessDeniedError when user lacks permission."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_delete_acl.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"], "teams": ["eng"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "title": "Secret"})

        # A different user with non-overlapping attributes should be denied
        other_user = User("bob", {"roles": ["viewer"], "teams": ["marketing"]})
        mock_get_authenticated_user.return_value = other_user

        with pytest.raises(AccessDeniedError):
            await sqlstore.delete("docs", where={"id": "doc1"})

        # Verify the row was not deleted
        mock_get_authenticated_user.return_value = owner
        row = await sqlstore.fetch_one("docs", where={"id": "doc1"})
        assert row is not None
        assert row["title"] == "Secret"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_delete_allowed_for_authorized_user(mock_get_authenticated_user):
    """Test that delete() succeeds when the user has matching attributes."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_delete_allowed.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "title": "To Delete"})

        # A user with matching roles can delete
        teammate = User("carol", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = teammate
        await sqlstore.delete("docs", where={"id": "doc1"})

        # Verify deletion
        mock_get_authenticated_user.return_value = owner
        row = await sqlstore.fetch_one("docs", where={"id": "doc1"})
        assert row is None


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_update_and_delete_allow_public_records(mock_get_authenticated_user):
    """Test that update() and delete() allow access to unowned (public) records."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_public_acl.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        # Insert a public record (no authenticated user)
        mock_get_authenticated_user.return_value = None
        await sqlstore.insert("docs", {"id": "pub1", "title": "Public"})

        # Any authenticated user should be able to update/delete public records
        any_user = User("anyone", {"roles": ["viewer"]})
        mock_get_authenticated_user.return_value = any_user
        await sqlstore.update("docs", {"title": "Updated Public"}, where={"id": "pub1"})

        raw = await base_sqlstore.fetch_all("docs", where={"id": "pub1"})
        assert raw.data[0]["title"] == "Updated Public"

        await sqlstore.delete("docs", where={"id": "pub1"})
        raw = await base_sqlstore.fetch_all("docs", where={"id": "pub1"})
        assert len(raw.data) == 0


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_update_with_access_control_fields_only_is_noop(mock_get_authenticated_user):
    """Test that update() safely no-ops when only ACL metadata fields are provided."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_update_noop.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "title": "Original"})

        await sqlstore.update(
            "docs",
            {"owner_principal": "mallory", "access_attributes": {"roles": ["viewer"]}},
            where={"id": "doc1"},
        )

        raw = await base_sqlstore.fetch_one("docs", where={"id": "doc1"})
        assert raw is not None
        assert raw["title"] == "Original"
        assert raw["owner_principal"] == "alice"
        assert raw["access_attributes"] == {"roles": ["admin"]}


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_update_race_does_not_modify_newly_inserted_unauthorized_rows(mock_get_authenticated_user):
    """Test that UPDATE ACL filtering also protects rows inserted after pre-check."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_update_race_guard.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "tag": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "tag": "target", "title": "Original"})

        updater = User("carol", {"roles": ["admin"]})
        intruder = User("bob", {"roles": ["viewer"]})
        mock_get_authenticated_user.return_value = updater

        real_update = base_sqlstore.update

        async def update_with_race(
            table: str,
            data: dict[str, str],
            where: dict[str, str],
            where_sql: str | None = None,
            where_sql_params: dict[str, str] | None = None,
        ) -> None:
            mock_get_authenticated_user.return_value = intruder
            await sqlstore.insert("docs", {"id": "doc2", "tag": "target", "title": "Secret"})
            mock_get_authenticated_user.return_value = updater
            await real_update(
                table,
                data,
                where,
                where_sql=where_sql,
                where_sql_params=where_sql_params,
            )

        with patch.object(base_sqlstore, "update", side_effect=update_with_race):
            await sqlstore.update("docs", {"title": "Updated"}, where={"tag": "target"})

        raw = await base_sqlstore.fetch_all("docs", where={"tag": "target"}, order_by=[("id", "asc")])
        assert len(raw.data) == 2
        assert raw.data[0]["id"] == "doc1"
        assert raw.data[0]["title"] == "Updated"
        assert raw.data[1]["id"] == "doc2"
        assert raw.data[1]["title"] == "Secret"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_invalid_attribute_keys_are_rejected(mock_get_authenticated_user):
    """Test that attribute keys containing non-alphanumeric characters are skipped during SQL filtering."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_invalid_keys.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "title": ColumnType.STRING},
        )

        victim = User("victim", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = victim
        await sqlstore.insert("docs", {"id": "doc1", "title": "Secret Document"})

        # Simulate an attacker whose attributes contain a malicious key
        attacker = User("attacker", {"') OR 1=1--": ["anything"]})
        mock_get_authenticated_user.return_value = attacker

        result = await sqlstore.fetch_all("docs")
        assert len(result.data) == 0, "Attacker with invalid attribute key must not see other users' documents"


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_json_extract_text_rejects_malicious_path(mock_get_authenticated_user):
    """Test that _json_extract_text raises ValueError for paths with special characters."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_json_path.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        with pytest.raises(ValueError, match="Invalid attribute key"):
            sqlstore._json_extract_text("access_attributes", "') OR 1=1--")

        with pytest.raises(ValueError, match="Invalid attribute key"):
            sqlstore._json_extract("access_attributes", "roles'; DROP TABLE docs;--")

        # Valid keys should work fine
        sqlstore._json_extract_text("access_attributes", "roles")
        sqlstore._json_extract_text("access_attributes", "teams")
        sqlstore._json_extract("access_attributes", "projects")
        sqlstore._json_extract("access_attributes", "namespaces")


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_delete_race_does_not_remove_newly_inserted_unauthorized_rows(mock_get_authenticated_user):
    """Test that DELETE ACL filtering also protects rows inserted after pre-check."""
    with TemporaryDirectory() as tmp_dir:
        base_sqlstore = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=tmp_dir + "/test_delete_race_guard.db"))
        sqlstore = AuthorizedSqlStore(base_sqlstore, default_policy())

        await sqlstore.create_table(
            table="docs",
            schema={"id": ColumnType.STRING, "tag": ColumnType.STRING, "title": ColumnType.STRING},
        )

        owner = User("alice", {"roles": ["admin"]})
        mock_get_authenticated_user.return_value = owner
        await sqlstore.insert("docs", {"id": "doc1", "tag": "target", "title": "Keep/Drop"})

        deleter = User("carol", {"roles": ["admin"]})
        intruder = User("bob", {"roles": ["viewer"]})
        mock_get_authenticated_user.return_value = deleter

        real_delete = base_sqlstore.delete

        async def delete_with_race(
            table: str,
            where: dict[str, str],
            where_sql: str | None = None,
            where_sql_params: dict[str, str] | None = None,
        ) -> None:
            mock_get_authenticated_user.return_value = intruder
            await sqlstore.insert("docs", {"id": "doc2", "tag": "target", "title": "Secret"})
            mock_get_authenticated_user.return_value = deleter
            await real_delete(
                table,
                where,
                where_sql=where_sql,
                where_sql_params=where_sql_params,
            )

        with patch.object(base_sqlstore, "delete", side_effect=delete_with_race):
            await sqlstore.delete("docs", where={"tag": "target"})

        raw = await base_sqlstore.fetch_all("docs", where={"tag": "target"}, order_by=[("id", "asc")])
        assert len(raw.data) == 1
        assert raw.data[0]["id"] == "doc2"
        assert raw.data[0]["title"] == "Secret"
