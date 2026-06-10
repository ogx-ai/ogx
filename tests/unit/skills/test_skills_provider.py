# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.inline.skills.builtin.config import BuiltinSkillsConfig
from ogx.providers.inline.skills.builtin.impl import BuiltinSkillsImpl
from ogx_api.files.models import OpenAIFileObject
from ogx_api.skills.models import (
    ListSkillsRequest,
    ListSkillVersionsRequest,
    SkillUpdateRequest,
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
def impl(kvstore, files_api):
    from ogx.core.storage.datatypes import KVStoreReference

    config = BuiltinSkillsConfig(persistence=KVStoreReference(backend="kv_default", namespace="skills"))
    return BuiltinSkillsImpl(config, files_api, kvstore)


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
