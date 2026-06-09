# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import json
import time
import uuid

from fastapi import Response, UploadFile

from ogx.core.storage.kvstore import KVStore
from ogx.log import get_logger
from ogx_api.files import (
    DeleteFileRequest,
    Files,
    OpenAIFileUploadPurpose,
    RetrieveFileContentRequest,
    UploadFileRequest,
)
from ogx_api.skills import (
    ListSkillsRequest,
    ListSkillsResponse,
    ListSkillVersionsRequest,
    ListSkillVersionsResponse,
    Skill,
    SkillDeleteResponse,
    Skills,
    SkillUpdateRequest,
    SkillVersion,
    SkillVersionCreateRequest,
    SkillVersionDeleteResponse,
)

from .config import BuiltinSkillsConfig
from .validation import validate_skill_zip

logger = get_logger(__name__)

_SKILL_PREFIX = "skill:"
_VERSION_PREFIX = "skill_version:"


def _skill_key(skill_id: str) -> str:
    return f"{_SKILL_PREFIX}{skill_id}"


def _version_key(skill_id: str, version: str) -> str:
    return f"{_VERSION_PREFIX}{skill_id}:{version}"


def _version_range_start(skill_id: str) -> str:
    return f"{_VERSION_PREFIX}{skill_id}:"


def _version_range_end(skill_id: str) -> str:
    return f"{_VERSION_PREFIX}{skill_id}:\xff"


def _new_skill_id() -> str:
    return f"skill-{uuid.uuid4().hex}"


def _new_version_id() -> str:
    return f"skillver-{uuid.uuid4().hex}"


class BuiltinSkillsImpl(Skills):
    """Built-in Skills provider backed by Files API for storage and KVStore for metadata."""

    def __init__(self, config: BuiltinSkillsConfig, files_api: Files, kvstore: KVStore):
        self.config = config
        self.files_api = files_api
        self.kvstore = kvstore

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def _store_bundle(self, content: bytes, filename: str) -> str:
        """Upload a zip bundle via the Files API and return the file ID."""
        upload_file = UploadFile(
            file=io.BytesIO(content),
            filename=filename,
            size=len(content),
        )
        request = UploadFileRequest(purpose=OpenAIFileUploadPurpose.ASSISTANTS)
        result = await self.files_api.openai_upload_file(request, upload_file)
        return result.id

    async def _delete_bundle(self, file_id: str) -> None:
        """Delete a zip bundle from the Files API."""
        try:
            await self.files_api.openai_delete_file(DeleteFileRequest(file_id=file_id))
        except Exception:
            logger.warning("Failed to delete file from storage", file_id=file_id)

    async def create_skill(self, file: UploadFile) -> Skill:
        content = await file.read()
        manifest, _ = validate_skill_zip(content)

        skill_id = _new_skill_id()
        now = int(time.time())

        file_id = await self._store_bundle(content, f"{skill_id}_v1.zip")

        version = SkillVersion(
            id=_new_version_id(),
            created_at=now,
            description=manifest.description or "",
            name=manifest.name or "",
            skill_id=skill_id,
            version="1",
        )
        await self.kvstore.set(_version_key(skill_id, "1"), version.model_dump_json())

        skill = Skill(
            id=skill_id,
            created_at=now,
            default_version="1",
            description=manifest.description or "",
            latest_version="1",
            name=manifest.name or "",
        )

        # Store file_id alongside skill metadata for content retrieval
        skill_data = skill.model_dump()
        skill_data["_file_ids"] = {"1": file_id}

        await self.kvstore.set(_skill_key(skill_id), json.dumps(skill_data))

        logger.info("Created skill", skill_id=skill_id, name=manifest.name)
        return skill

    async def list_skills(self, request: ListSkillsRequest) -> ListSkillsResponse:
        values = await self.kvstore.values_in_range(_SKILL_PREFIX, f"{_SKILL_PREFIX}\xff")

        skills = [Skill(**{k: v for k, v in json.loads(v).items() if not k.startswith("_")}) for v in values]

        if request.order == "asc":
            skills.sort(key=lambda s: s.created_at)
        else:
            skills.sort(key=lambda s: s.created_at, reverse=True)

        # Apply cursor pagination
        start_idx = 0
        if request.after:
            for i, s in enumerate(skills):
                if s.id == request.after:
                    start_idx = i + 1
                    break

        page = skills[start_idx : start_idx + request.limit]
        has_more = start_idx + request.limit < len(skills)

        return ListSkillsResponse(
            data=page,
            has_more=has_more,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
        )

    async def get_skill(self, skill_id: str) -> Skill:
        data = await self.kvstore.get(_skill_key(skill_id))
        if data is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")

        raw = json.loads(data)
        return Skill(**{k: v for k, v in raw.items() if not k.startswith("_")})

    async def update_skill(self, skill_id: str, request: SkillUpdateRequest) -> Skill:
        data = await self.kvstore.get(_skill_key(skill_id))
        if data is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")

        raw = json.loads(data)

        version_data = await self.kvstore.get(_version_key(skill_id, request.default_version))
        if version_data is None:
            raise ValueError(f"Failed to update skill: version '{request.default_version}' does not exist")

        raw["default_version"] = request.default_version
        await self.kvstore.set(_skill_key(skill_id), json.dumps(raw))

        logger.info("Updated skill default version", skill_id=skill_id, version=request.default_version)
        return Skill(**{k: v for k, v in raw.items() if not k.startswith("_")})

    async def delete_skill(self, skill_id: str) -> SkillDeleteResponse:
        data = await self.kvstore.get(_skill_key(skill_id))
        if data is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")

        raw = json.loads(data)
        file_ids: dict[str, str] = raw.get("_file_ids", {})

        # Delete all version bundles and metadata
        for fid in file_ids.values():
            await self._delete_bundle(fid)

        version_values = await self.kvstore.values_in_range(
            _version_range_start(skill_id), _version_range_end(skill_id)
        )
        for v_data in version_values:
            ver = SkillVersion.model_validate_json(v_data)
            await self.kvstore.delete(_version_key(skill_id, ver.version))

        await self.kvstore.delete(_skill_key(skill_id))

        logger.info("Deleted skill", skill_id=skill_id)
        return SkillDeleteResponse(id=skill_id)

    async def get_skill_content(self, skill_id: str) -> Response:
        skill = await self.get_skill(skill_id)
        return await self.get_skill_version_content(skill_id, skill.default_version)

    async def create_skill_version(
        self, skill_id: str, request: SkillVersionCreateRequest, file: UploadFile
    ) -> SkillVersion:
        data = await self.kvstore.get(_skill_key(skill_id))
        if data is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")

        raw = json.loads(data)
        content = await file.read()
        manifest, _ = validate_skill_zip(content)

        next_version = str(int(raw["latest_version"]) + 1)
        now = int(time.time())

        file_id = await self._store_bundle(content, f"{skill_id}_v{next_version}.zip")

        version = SkillVersion(
            id=_new_version_id(),
            created_at=now,
            description=manifest.description or "",
            name=manifest.name or "",
            skill_id=skill_id,
            version=next_version,
        )
        await self.kvstore.set(_version_key(skill_id, next_version), version.model_dump_json())

        raw["latest_version"] = next_version
        file_ids: dict[str, str] = raw.get("_file_ids", {})
        file_ids[next_version] = file_id
        raw["_file_ids"] = file_ids

        if request.default:
            raw["default_version"] = next_version

        await self.kvstore.set(_skill_key(skill_id), json.dumps(raw))

        logger.info("Created skill version", skill_id=skill_id, version=next_version)
        return version

    async def list_skill_versions(self, skill_id: str, request: ListSkillVersionsRequest) -> ListSkillVersionsResponse:
        await self.get_skill(skill_id)

        values = await self.kvstore.values_in_range(_version_range_start(skill_id), _version_range_end(skill_id))
        versions = [SkillVersion.model_validate_json(v) for v in values]

        if request.order == "asc":
            versions.sort(key=lambda v: int(v.version))
        else:
            versions.sort(key=lambda v: int(v.version), reverse=True)

        start_idx = 0
        if request.after:
            for i, v in enumerate(versions):
                if v.id == request.after:
                    start_idx = i + 1
                    break

        page = versions[start_idx : start_idx + request.limit]
        has_more = start_idx + request.limit < len(versions)

        return ListSkillVersionsResponse(
            data=page,
            has_more=has_more,
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
        )

    async def get_skill_version(self, skill_id: str, version: str) -> SkillVersion:
        await self.get_skill(skill_id)

        data = await self.kvstore.get(_version_key(skill_id, version))
        if data is None:
            raise ValueError(f"Failed to find skill version: '{skill_id}' version '{version}' does not exist")
        return SkillVersion.model_validate_json(data)

    async def get_skill_version_content(self, skill_id: str, version: str) -> Response:
        data = await self.kvstore.get(_skill_key(skill_id))
        if data is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")

        raw = json.loads(data)
        file_ids: dict[str, str] = raw.get("_file_ids", {})
        file_id = file_ids.get(version)
        if file_id is None:
            raise ValueError(f"Failed to retrieve skill content: no bundle stored for '{skill_id}' version '{version}'")

        resp = await self.files_api.openai_retrieve_file_content(RetrieveFileContentRequest(file_id=file_id))
        return Response(
            content=resp.body,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{skill_id}_v{version}.zip"'},
        )

    async def delete_skill_version(self, skill_id: str, version: str) -> SkillVersionDeleteResponse:
        data = await self.kvstore.get(_skill_key(skill_id))
        if data is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")

        raw = json.loads(data)

        version_data = await self.kvstore.get(_version_key(skill_id, version))
        if version_data is None:
            raise ValueError(f"Failed to find skill version: '{skill_id}' version '{version}' does not exist")

        if raw["default_version"] == version and raw["latest_version"] == version:
            # Check if this is the only version
            all_versions = await self.kvstore.values_in_range(
                _version_range_start(skill_id), _version_range_end(skill_id)
            )
            if len(all_versions) <= 1:
                raise ValueError(
                    "Failed to delete skill version: cannot delete the only version. Delete the skill instead."
                )

        file_ids: dict[str, str] = raw.get("_file_ids", {})
        file_id = file_ids.pop(version, None)
        if file_id:
            await self._delete_bundle(file_id)

        await self.kvstore.delete(_version_key(skill_id, version))

        raw["_file_ids"] = file_ids
        await self.kvstore.set(_skill_key(skill_id), json.dumps(raw))

        logger.info("Deleted skill version", skill_id=skill_id, version=version)
        return SkillVersionDeleteResponse(id=skill_id, version=version)
