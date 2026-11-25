# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import io
import json
import time
import uuid
from typing import cast

from fastapi import Response, UploadFile

from ogx.core.datatypes import TenancyMode
from ogx.core.storage.kvstore import KVStore
from ogx.core.storage.sqlstore.authorized_sqlstore import (
    AuthorizedSqlStore,
    _enhance_item_with_access_control,
)
from ogx.log import get_logger
from ogx.providers.utils.files.response import response_body_bytes
from ogx_api.files import (
    DeleteFileRequest,
    Files,
    OpenAIFileUploadPurpose,
    RetrieveFileContentRequest,
    UploadFileRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, DeleteOperation
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

_TABLE_SKILLS = "skills"
_TABLE_SKILL_VERSIONS = "skill_versions"
# Legacy key prefixes from the pre-authorization implementation, where all skill
# metadata lived in the KV store with no owner information. Retained so the
# one-way startup migration can find and relocate those rows.
_SKILL_PREFIX = "skill:"
_VERSION_PREFIX = "skill_version:"
_FILE_IDS_PREFIX = "skill_files:"


def _new_skill_id() -> str:
    return f"skill-{uuid.uuid4().hex}"


def _new_version_id() -> str:
    return f"skillver-{uuid.uuid4().hex}"


class BuiltinSkillsImpl(Skills):
    """Built-in Skills provider backed by the Files API for bundle storage and an
    AuthorizedSqlStore for metadata, so every skill row carries ownership and
    tenant attributes and is scoped by the configured access policy."""

    def __init__(self, config: BuiltinSkillsConfig, files_api: Files, kvstore: KVStore, sql_store: AuthorizedSqlStore):
        self.config = config
        self.files_api = files_api
        self.kvstore = kvstore
        self.sql_store = sql_store
        # Serializes the read-compute-insert sequence in create_skill_version so
        # two concurrent version creates cannot both compute the same next version
        # number and both insert it.
        self._version_lock = asyncio.Lock()

    async def initialize(self) -> None:
        await self.sql_store.create_table(
            _TABLE_SKILLS,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.STRING,
                "skill_data": ColumnType.JSON,
            },
        )
        await self.sql_store.create_table(
            _TABLE_SKILL_VERSIONS,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "skill_id": ColumnType.STRING,
                "version": ColumnType.STRING,
                "created_at": ColumnType.STRING,
                "file_id": ColumnType.STRING,
                "version_data": ColumnType.JSON,
            },
        )
        await self._migrate_legacy_kv_skills()

    async def _migrate_legacy_kv_skills(self) -> None:
        """Relocate skill metadata written by the pre-authorization implementation.

        Before skills moved to the SQL store, all metadata (skills, versions, and
        the per-version bundle file-id map) lived in the KV store under unowned
        keys, so existing deployments have rows the new storage path cannot see.
        Re-inserting them on startup keeps previously created skills readable
        after an upgrade. The legacy rows carry no owner information; inserting
        them without a request context marks them as server-owned rows
        (owner_principal=""), which under the default policy preserves exactly
        the pre-upgrade visibility. The per-row existence check keeps the
        migration a no-op on later startups, and the insert is conflict-tolerant
        so several server processes sharing the same storage (e.g. a
        multi-replica deployment) can run it concurrently without racing on the
        primary key. The legacy KV entries are left in place so the data remains
        recoverable.
        """
        file_ids_by_skill: dict[str, dict[str, str]] = {}
        for key in await self.kvstore.keys_in_range(f"{_FILE_IDS_PREFIX}", f"{_FILE_IDS_PREFIX}\xff"):
            raw = await self.kvstore.get(key)
            if raw is None:
                continue
            try:
                file_ids_by_skill[key.removeprefix(f"{_FILE_IDS_PREFIX}")] = _parse_file_ids(raw)
            except Exception:
                logger.warning("Skipping unparseable legacy skill file-ids entry during migration")

        migrated = 0
        legacy_values = await self.kvstore.values_in_range(f"{_SKILL_PREFIX}", f"{_SKILL_PREFIX}\xff")
        for value in legacy_values:
            try:
                skill = Skill.model_validate_json(value)
            except Exception:
                logger.warning("Skipping unparseable legacy skill entry during migration")
                continue
            # Bootstrap-time check: the migration runs outside any request context, so
            # it must see rows of every tenant to stay a no-op in multi-tenant mode.
            existing = await self.sql_store.sql_store.fetch_all(table=_TABLE_SKILLS, where={"id": skill.id}, limit=1)
            if existing.data:
                continue
            row = _enhance_item_with_access_control(
                {
                    "id": skill.id,
                    "created_at": str(skill.created_at),
                    "skill_data": skill.model_dump(),
                },
                None,
                self.sql_store.tenancy_mode,
                self.sql_store.default_tenant_id,
            )
            await self.sql_store.sql_store.insert_do_nothing(table=_TABLE_SKILLS, data=row)
            migrated += 1

        migrated_versions = 0
        version_values = await self.kvstore.values_in_range(f"{_VERSION_PREFIX}", f"{_VERSION_PREFIX}\xff")
        for value in version_values:
            try:
                version = SkillVersion.model_validate_json(value)
            except Exception:
                logger.warning("Skipping unparseable legacy skill version entry during migration")
                continue
            existing = await self.sql_store.sql_store.fetch_all(
                table=_TABLE_SKILL_VERSIONS, where={"id": version.id}, limit=1
            )
            if existing.data:
                continue
            row = _enhance_item_with_access_control(
                {
                    "id": version.id,
                    "skill_id": version.skill_id,
                    "version": version.version,
                    "created_at": str(version.created_at),
                    "file_id": file_ids_by_skill.get(version.skill_id, {}).get(version.version),
                    "version_data": version.model_dump(),
                },
                None,
                self.sql_store.tenancy_mode,
                self.sql_store.default_tenant_id,
            )
            await self.sql_store.sql_store.insert_do_nothing(table=_TABLE_SKILL_VERSIONS, data=row)
            migrated_versions += 1

        if migrated or migrated_versions:
            if self.sql_store.tenancy_mode == TenancyMode.MULTI:
                # In multi-tenant mode there is no tenant to attribute the legacy rows
                # to; they are kept in the table but invisible to tenant-scoped users.
                logger.warning(
                    "Migrated legacy skills without owner or tenant information; "
                    "they are not visible to tenant-scoped users in multi-tenant mode",
                    skills=migrated,
                    versions=migrated_versions,
                )
            else:
                logger.info(
                    "Migrated legacy skills from KV store",
                    skills=migrated,
                    versions=migrated_versions,
                )

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
        """Delete a zip bundle from the Files API.

        A failure here is not swallowed: the delete request must fail rather
        than drop the metadata and orphan the bundle.
        """
        await self.files_api.openai_delete_file(DeleteFileRequest(file_id=file_id))

    async def _load_skill(self, skill_id: str) -> Skill:
        record = await self.sql_store.fetch_one(table=_TABLE_SKILLS, where={"id": skill_id})
        if record is None:
            raise ValueError(f"Failed to find skill: '{skill_id}' does not exist")
        return Skill.model_validate(record["skill_data"])

    async def _save_skill(self, skill: Skill) -> None:
        await self.sql_store.update(
            table=_TABLE_SKILLS,
            data={"skill_data": skill.model_dump()},
            where={"id": skill.id},
        )

    async def _load_version(self, skill_id: str, version: str) -> SkillVersion:
        record = await self.sql_store.fetch_one(
            table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id, "version": version}
        )
        if record is None:
            raise ValueError(f"Failed to find skill version: '{skill_id}' version '{version}' does not exist")
        return SkillVersion.model_validate(record["version_data"])

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
        skill = Skill(
            id=skill_id,
            created_at=now,
            default_version="1",
            description=manifest.description or "",
            latest_version="1",
            name=manifest.name or "",
        )

        await self.sql_store.insert(
            table=_TABLE_SKILLS,
            data={
                "id": skill_id,
                "created_at": str(now),
                "skill_data": skill.model_dump(),
            },
        )
        await self.sql_store.insert(
            table=_TABLE_SKILL_VERSIONS,
            data={
                "id": version.id,
                "skill_id": skill_id,
                "version": version.version,
                "created_at": str(now),
                "file_id": file_id,
                "version_data": version.model_dump(),
            },
        )

        logger.info("Created skill", skill_id=skill_id, name=manifest.name)
        return skill

    async def list_skills(self, request: ListSkillsRequest) -> ListSkillsResponse:
        # Filtering is delegated to AuthorizedSqlStore.fetch_all, which enforces the
        # configured access policy at the row level -- this returns only skills the
        # caller owns (or is otherwise permitted to see under that policy), not all
        # skills in the table.
        results = await self.sql_store.fetch_all(table=_TABLE_SKILLS)

        skills = [Skill.model_validate(row["skill_data"]) for row in results.data]

        if request.order == "asc":
            skills.sort(key=lambda s: s.created_at)
        else:
            skills.sort(key=lambda s: s.created_at, reverse=True)

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
        return await self._load_skill(skill_id)

    async def update_skill(self, skill_id: str, request: SkillUpdateRequest) -> Skill:
        skill = await self._load_skill(skill_id)

        await self._load_version(skill_id, request.default_version)

        skill.default_version = request.default_version
        await self._save_skill(skill)

        logger.info("Updated skill default version", skill_id=skill_id, version=request.default_version)
        return skill

    async def delete_skill(self, skill_id: str) -> SkillDeleteResponse:
        await self._load_skill(skill_id)

        versions = await self.sql_store.fetch_all(table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id})
        for row in versions.data:
            if row.get("file_id"):
                await self._delete_bundle(row["file_id"])

        await self.sql_store.delete_many(
            [
                DeleteOperation(table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id}),
                DeleteOperation(table=_TABLE_SKILLS, where={"id": skill_id}),
            ]
        )

        logger.info("Deleted skill", skill_id=skill_id)
        return SkillDeleteResponse(id=skill_id)

    async def get_skill_content(self, skill_id: str) -> Response:
        skill = await self._load_skill(skill_id)
        return await self.get_skill_version_content(skill_id, skill.default_version)

    async def create_skill_version(
        self, skill_id: str, request: SkillVersionCreateRequest, file: UploadFile
    ) -> SkillVersion:
        skill = await self._load_skill(skill_id)
        content = await file.read()
        manifest, _ = validate_skill_zip(content)

        async with self._version_lock:
            # Re-read under the lock: latest_version may have advanced between the
            # initial read and the insert (concurrent create_skill_version calls).
            skill = await self._load_skill(skill_id)
            next_version = str(int(skill.latest_version) + 1)
            existing = await self.sql_store.fetch_one(
                table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id, "version": next_version}
            )
            if existing is not None:
                raise ValueError(
                    f"Failed to create skill version: version '{next_version}' already exists for skill '{skill_id}'"
                )
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
            await self.sql_store.insert(
                table=_TABLE_SKILL_VERSIONS,
                data={
                    "id": version.id,
                    "skill_id": skill_id,
                    "version": next_version,
                    "created_at": str(now),
                    "file_id": file_id,
                    "version_data": version.model_dump(),
                },
            )

            skill.latest_version = next_version
            if request.default:
                skill.default_version = next_version
            await self._save_skill(skill)

        logger.info("Created skill version", skill_id=skill_id, version=next_version)
        return version

    async def list_skill_versions(self, skill_id: str, request: ListSkillVersionsRequest) -> ListSkillVersionsResponse:
        await self._load_skill(skill_id)

        results = await self.sql_store.fetch_all(table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id})
        versions = [SkillVersion.model_validate(row["version_data"]) for row in results.data]

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
        await self._load_skill(skill_id)

        return await self._load_version(skill_id, version)

    async def get_skill_version_content(self, skill_id: str, version: str) -> Response:
        await self._load_skill(skill_id)

        record = await self.sql_store.fetch_one(
            table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id, "version": version}
        )
        file_id = record.get("file_id") if record else None
        if file_id is None:
            raise ValueError(f"Failed to retrieve skill content: no bundle stored for '{skill_id}' version '{version}'")

        resp = await self.files_api.openai_retrieve_file_content(RetrieveFileContentRequest(file_id=file_id))
        return Response(
            content=await response_body_bytes(resp),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{skill_id}_v{version}.zip"'},
        )

    async def delete_skill_version(self, skill_id: str, version: str) -> SkillVersionDeleteResponse:
        skill = await self._load_skill(skill_id)

        target = await self.sql_store.fetch_one(
            table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id, "version": version}
        )
        if target is None:
            raise ValueError(f"Failed to find skill version: '{skill_id}' version '{version}' does not exist")

        all_versions = await self.sql_store.fetch_all(table=_TABLE_SKILL_VERSIONS, where={"skill_id": skill_id})
        if len(all_versions.data) <= 1:
            raise ValueError(
                "Failed to delete skill version: cannot delete the only version. Delete the skill instead."
            )

        file_id = target.get("file_id")
        if file_id:
            await self._delete_bundle(file_id)

        await self.sql_store.delete(table=_TABLE_SKILL_VERSIONS, where={"id": target["id"]})

        remaining = [row for row in all_versions.data if row["version"] != version]
        remaining_versions = [SkillVersion.model_validate(row["version_data"]) for row in remaining]
        remaining_versions.sort(key=lambda v: int(v.version), reverse=True)

        # Update default_version if the deleted version was the default
        if skill.default_version == version:
            skill.default_version = remaining_versions[0].version

        # Update latest_version if the deleted version was the latest
        if skill.latest_version == version:
            skill.latest_version = remaining_versions[0].version

        await self._save_skill(skill)

        logger.info("Deleted skill version", skill_id=skill_id, version=version)
        return SkillVersionDeleteResponse(id=skill_id, version=version)


def _parse_file_ids(raw: str) -> dict[str, str]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("legacy skill file-ids entry is not an object")
    return cast(dict[str, str], parsed)
