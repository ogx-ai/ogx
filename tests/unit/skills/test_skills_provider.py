# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.datatypes import User
from ogx.core.request_headers import RequestProviderDataContext
from ogx.core.storage.datatypes import KVStoreReference, SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import (
    _sqlstore_impl,
    register_sqlstore_backends,
    shutdown_sqlstore_backends,
)
from ogx.providers.inline.skills.builtin.config import BuiltinSkillsConfig
from ogx.providers.inline.skills.builtin.impl import BuiltinSkillsImpl
from ogx_api.files.models import OpenAIFileObject
from ogx_api.skills.models import (
    ListSkillsRequest,
    ListSkillVersionsRequest,
    Skill,
    SkillUpdateRequest,
    SkillVersion,
    SkillVersionCreateRequest,
)


def _make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


SKILL_MD = """---
name: test-skill
description: A test skill for unit tests
---
Use this skill for testing."""


def _make_upload_file(content: bytes, filename: str = "skill.zip"):
    from fastapi import UploadFile

    return UploadFile(file=io.BytesIO(content), filename=filename, size=len(content))


def _make_file_object(file_id: str) -> OpenAIFileObject:
    return OpenAIFileObject(
        id=file_id,
        bytes=100,
        created_at=1000,
        filename="test.zip",
        object="file",
        purpose="assistants",
        status="processed",
    )


class _FakeKVStore:
    """In-memory KVStore for testing."""

    def __init__(self):
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, expiration=None) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [k for k in sorted(self._data.keys()) if start_key <= k < end_key]

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [self._data[k] for k in sorted(self._data.keys()) if start_key <= k < end_key]


@pytest.fixture
def kvstore():
    return _FakeKVStore()


@pytest.fixture
def files_api():
    api = AsyncMock()
    api.openai_upload_file = AsyncMock(return_value=_make_file_object("file-abc123"))
    api.openai_delete_file = AsyncMock()
    api.openai_retrieve_file_content = AsyncMock(return_value=MagicMock(body=b"zip-content"))
    return api


@pytest.fixture
async def impl(kvstore, files_api):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_skills.db"
        backend_name = "sql_skills_test"
        register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=str(db_path))})
        config = BuiltinSkillsConfig(
            sqlstore=SqlStoreReference(backend=backend_name, table_name="skills"),
            persistence=KVStoreReference(backend="kv_default", namespace="skills"),
        )

        base_sql_store = await _sqlstore_impl(config.sqlstore)
        sql_store = AuthorizedSqlStore(base_sql_store, policy=[])

        impl = BuiltinSkillsImpl(config, files_api, kvstore, sql_store)
        await impl.initialize()

        yield impl

        await impl.shutdown()
        # Dispose the engine while the tempdir still exists; leaving pooled
        # connections open over a deleted sqlite file makes later tests flaky.
        await shutdown_sqlstore_backends()


class TestCreateSkill:
    async def test_create_skill(self, impl, files_api):
        content = _make_zip({"SKILL.md": SKILL_MD})
        upload = _make_upload_file(content)

        skill = await impl.create_skill(upload)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit tests"
        assert skill.default_version == "1"
        assert skill.latest_version == "1"
        assert skill.object == "skill"
        assert skill.id.startswith("skill-")
        files_api.openai_upload_file.assert_called_once()

    async def test_create_skill_invalid_zip(self, impl):
        upload = _make_upload_file(b"not a zip")
        with pytest.raises(ValueError, match="not a valid zip archive"):
            await impl.create_skill(upload)


class TestGetSkill:
    async def test_get_skill(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        created = await impl.create_skill(_make_upload_file(content))

        fetched = await impl.get_skill(created.id)
        assert fetched.id == created.id
        assert fetched.name == "test-skill"

    async def test_get_nonexistent_skill(self, impl):
        with pytest.raises(ValueError, match="does not exist"):
            await impl.get_skill("skill-nonexistent")


class TestListSkills:
    async def test_list_skills(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        await impl.create_skill(_make_upload_file(content))
        await impl.create_skill(_make_upload_file(content))

        result = await impl.list_skills(ListSkillsRequest())
        assert len(result.data) == 2
        assert result.object == "list"

    async def test_list_skills_empty(self, impl):
        result = await impl.list_skills(ListSkillsRequest())
        assert len(result.data) == 0
        assert not result.has_more

    async def test_list_skills_pagination(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        for _ in range(3):
            await impl.create_skill(_make_upload_file(content))

        result = await impl.list_skills(ListSkillsRequest(limit=2))
        assert len(result.data) == 2
        assert result.has_more


class TestUpdateSkill:
    async def test_update_default_version(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        updated = await impl.update_skill(skill.id, SkillUpdateRequest(default_version="2"))
        assert updated.default_version == "2"

    async def test_update_nonexistent_version(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        with pytest.raises(ValueError, match="does not exist"):
            await impl.update_skill(skill.id, SkillUpdateRequest(default_version="99"))


class TestDeleteSkill:
    async def test_delete_skill(self, impl, files_api):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        result = await impl.delete_skill(skill.id)
        assert result.deleted is True
        assert result.id == skill.id
        assert result.object == "skill.deleted"

        with pytest.raises(ValueError, match="does not exist"):
            await impl.get_skill(skill.id)


class TestCreateSkillVersion:
    async def test_create_version(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        version = await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))
        assert version.version == "2"
        assert version.skill_id == skill.id

        updated_skill = await impl.get_skill(skill.id)
        assert updated_skill.latest_version == "2"
        assert updated_skill.default_version == "1"

    async def test_create_version_set_default(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        await impl.create_skill_version(
            skill.id,
            SkillVersionCreateRequest(default=True),
            _make_upload_file(content),
        )

        updated = await impl.get_skill(skill.id)
        assert updated.default_version == "2"


class TestListSkillVersions:
    async def test_list_versions(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        result = await impl.list_skill_versions(skill.id, ListSkillVersionsRequest())
        assert len(result.data) == 2


class TestGetSkillVersion:
    async def test_get_version(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        version = await impl.get_skill_version(skill.id, "1")
        assert version.version == "1"
        assert version.skill_id == skill.id

    async def test_get_nonexistent_version(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        with pytest.raises(ValueError, match="does not exist"):
            await impl.get_skill_version(skill.id, "99")


class TestDeleteSkillVersion:
    async def test_delete_version(self, impl, files_api):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        result = await impl.delete_skill_version(skill.id, "1")
        assert result.deleted is True
        assert result.version == "1"
        assert result.object == "skill.version.deleted"

    async def test_delete_default_version_updates_default(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        await impl.update_skill(skill.id, SkillUpdateRequest(default_version="2"))
        await impl.delete_skill_version(skill.id, "2")

        updated = await impl.get_skill(skill.id)
        assert updated.default_version != "2"
        assert updated.default_version in ("1", "3")

    async def test_delete_latest_version_updates_latest(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        await impl.delete_skill_version(skill.id, "2")

        updated = await impl.get_skill(skill.id)
        assert updated.latest_version == "1"

    async def test_delete_only_version(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        with pytest.raises(ValueError, match="cannot delete the only version"):
            await impl.delete_skill_version(skill.id, "1")


class TestGetSkillContent:
    async def test_get_content(self, impl, files_api):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        response = await impl.get_skill_content(skill.id)
        assert response.media_type == "application/zip"
        files_api.openai_retrieve_file_content.assert_called_once()


class TestSkillAuthorization:
    """Per-user scoping of skill metadata (mirrors the message batch
    authorization tests): skill metadata lives in an AuthorizedSqlStore, so
    list/get/update/version/create/delete are scoped to the creating user."""

    @staticmethod
    def _zip():
        return _make_zip({"SKILL.md": SKILL_MD})

    async def test_list_skills_scoped_to_creating_user(self, impl):
        """list_skills should only return skills owned by the authenticated
        caller, not every skill in the store (see AuthorizedSqlStore.fetch_all)."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            alice_skill = await impl.create_skill(_make_upload_file(self._zip()))
        with RequestProviderDataContext(user=bob):
            bob_skill = await impl.create_skill(_make_upload_file(self._zip()))

        with RequestProviderDataContext(user=alice):
            alice_list = await impl.list_skills(ListSkillsRequest())
        alice_ids = {s.id for s in alice_list.data}
        assert alice_skill.id in alice_ids
        assert bob_skill.id not in alice_ids

        with RequestProviderDataContext(user=bob):
            bob_list = await impl.list_skills(ListSkillsRequest())
        bob_ids = {s.id for s in bob_list.data}
        assert bob_skill.id in bob_ids
        assert alice_skill.id not in bob_ids

    async def test_get_skill_denied_for_other_user(self, impl):
        """A user cannot retrieve a skill created by a different user."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            skill = await impl.create_skill(_make_upload_file(self._zip()))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(ValueError, match="does not exist"):
                await impl.get_skill(skill.id)

        with RequestProviderDataContext(user=alice):
            retrieved = await impl.get_skill(skill.id)
        assert retrieved.id == skill.id

    async def test_update_skill_denied_for_other_user(self, impl):
        """A user cannot update a skill created by a different user."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            skill = await impl.create_skill(_make_upload_file(self._zip()))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(ValueError, match="does not exist"):
                await impl.update_skill(skill.id, SkillUpdateRequest(default_version="1"))

        with RequestProviderDataContext(user=alice):
            updated = await impl.update_skill(skill.id, SkillUpdateRequest(default_version="1"))
        assert updated.default_version == "1"

    async def test_create_skill_version_denied_for_other_user(self, impl):
        """A user cannot add a version (or hijack the default version) of
        another user's skill."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            skill = await impl.create_skill(_make_upload_file(self._zip()))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(ValueError, match="does not exist"):
                await impl.create_skill_version(
                    skill.id, SkillVersionCreateRequest(default=True), _make_upload_file(self._zip())
                )

        with RequestProviderDataContext(user=alice):
            unchanged = await impl.get_skill(skill.id)
        assert unchanged.latest_version == "1"
        assert unchanged.default_version == "1"

    async def test_get_skill_version_denied_for_other_user(self, impl):
        """A user cannot read another user's skill version metadata."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            skill = await impl.create_skill(_make_upload_file(self._zip()))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(ValueError, match="does not exist"):
                await impl.get_skill_version(skill.id, "1")

    async def test_delete_skill_denied_for_other_user(self, impl):
        """A user cannot delete a skill created by a different user."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            skill = await impl.create_skill(_make_upload_file(self._zip()))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(ValueError, match="does not exist"):
                await impl.delete_skill(skill.id)

        with RequestProviderDataContext(user=alice):
            still_there = await impl.get_skill(skill.id)
        assert still_there.id == skill.id


class TestLegacyMigration:
    """One-way startup migration of pre-authorization KV metadata."""

    @staticmethod
    async def _impl_with_legacy_kv(kvstore, files_api):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_skills_migrate.db"
            backend_name = "sql_skills_migrate_test"
            register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=str(db_path))})
            config = BuiltinSkillsConfig(
                sqlstore=SqlStoreReference(backend=backend_name, table_name="skills"),
                persistence=KVStoreReference(backend="kv_default", namespace="skills"),
            )
            base_sql_store = await _sqlstore_impl(config.sqlstore)
            sql_store = AuthorizedSqlStore(base_sql_store, policy=[])
            impl = BuiltinSkillsImpl(config, files_api, kvstore, sql_store)
            await impl.initialize()
            try:
                yield impl
            finally:
                await impl.shutdown()
                await shutdown_sqlstore_backends()

    @pytest.fixture
    async def migrated_impl(self, kvstore, files_api):
        skill = Skill(
            id="skill-legacy1",
            created_at=1000,
            default_version="1",
            description="legacy skill",
            latest_version="1",
            name="legacy",
        )
        version = SkillVersion(
            id="skillver-legacy1",
            created_at=1000,
            description="legacy skill",
            name="legacy",
            skill_id="skill-legacy1",
            version="1",
        )
        kvstore._data["skill:skill-legacy1"] = skill.model_dump_json()
        kvstore._data["skill_version:skill-legacy1:1"] = version.model_dump_json()
        kvstore._data["skill_files:skill-legacy1"] = '{"1": "file-legacy"}'

        async for impl in self._impl_with_legacy_kv(kvstore, files_api):
            yield impl

    async def test_legacy_skills_migrated_and_readable(self, migrated_impl):
        """Legacy KV rows are relocated to the SQL store and stay readable."""
        skill = await migrated_impl.get_skill("skill-legacy1")
        assert skill.name == "legacy"

        version = await migrated_impl.get_skill_version("skill-legacy1", "1")
        assert version.id == "skillver-legacy1"

        # The migrated version row carries the legacy file-id mapping.
        record = await migrated_impl.sql_store.fetch_one(table="skill_versions", where={"id": "skillver-legacy1"})
        assert record["file_id"] == "file-legacy"

    async def test_migration_is_idempotent(self, migrated_impl, kvstore):
        """A second startup against the same store does not duplicate rows."""
        # Simulate a second process startup over the same database: create a new
        # AuthorizedSqlStore on the same sqlite file and run the migration again.
        # (The fixture already ran it once at initialize time.)
        first = await migrated_impl.list_skills(ListSkillsRequest())
        assert len(first.data) == 1

        # Re-running the migration directly must be a no-op.
        await migrated_impl._migrate_legacy_kv_skills()
        second = await migrated_impl.list_skills(ListSkillsRequest())
        assert len(second.data) == 1
        versions = await migrated_impl.list_skill_versions("skill-legacy1", ListSkillVersionsRequest())
        assert len(versions.data) == 1

    async def test_migration_skips_unparseable_entries(self, kvstore, files_api):
        kvstore._data["skill:skill-bad"] = "not-json"
        kvstore._data["skill_version:skill-bad:1"] = "not-json"
        kvstore._data["skill_files:skill-bad"] = "not-json"

        async for impl in self._impl_with_legacy_kv(kvstore, files_api):
            skills = await impl.list_skills(ListSkillsRequest())
            assert skills.data == []


class TestDeleteBundleErrorHandling:
    """_delete_bundle must not swallow Files delete failures."""

    async def test_delete_skill_fails_when_bundle_delete_fails(self, impl, files_api):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        files_api.openai_delete_file = AsyncMock(side_effect=RuntimeError("storage down"))

        with pytest.raises(RuntimeError, match="storage down"):
            await impl.delete_skill(skill.id)

        # The skill metadata is untouched: the delete must not orphan the bundle.
        assert (await impl.get_skill(skill.id)).id == skill.id

    async def test_delete_version_fails_when_bundle_delete_fails(self, impl, files_api):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        files_api.openai_delete_file = AsyncMock(side_effect=RuntimeError("storage down"))

        with pytest.raises(RuntimeError, match="storage down"):
            await impl.delete_skill_version(skill.id, "2")

        # Version "2" metadata is still present.
        assert (await impl.get_skill_version(skill.id, "2")).id


class TestCreateSkillVersionConcurrency:
    """The read-compute-insert sequence must not produce duplicate version numbers
    under concurrent creates (the KV implementation overwrote the shared key, so a
    duplicate is new state to the SQL storage)."""

    async def test_concurrent_creates_do_not_duplicate_versions(self, impl):
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))

        created = await asyncio.gather(
            impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content)),
            impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content)),
        )
        assert {v.version for v in created} == {"2", "3"}

        listed = await impl.list_skill_versions(skill.id, ListSkillVersionsRequest())
        assert sorted(v.version for v in listed.data) == ["1", "2", "3"]
        assert (await impl.get_skill(skill.id)).latest_version == "3"

    async def test_create_version_rejects_existing_version_number(self, impl):
        """The pre-check rejects a version number that already has a row (e.g. a
        stale reader) instead of inserting a duplicate row."""
        content = _make_zip({"SKILL.md": SKILL_MD})
        skill = await impl.create_skill(_make_upload_file(content))
        await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        # Simulate a stale skill row: version "2" exists but latest_version says "1".
        record = await impl.sql_store.fetch_one(table="skills", where={"id": skill.id})
        stale = Skill.model_validate(record["skill_data"])
        stale.latest_version = "1"
        await impl._save_skill(stale)

        with pytest.raises(ValueError, match="already exists"):
            await impl.create_skill_version(skill.id, SkillVersionCreateRequest(), _make_upload_file(content))

        rows = await impl.sql_store.fetch_all(table="skill_versions", where={"skill_id": skill.id})
        assert sorted(r["version"] for r in rows.data) == ["1", "2"]
