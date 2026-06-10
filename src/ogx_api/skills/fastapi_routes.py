# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile
from fastapi.param_functions import File, Form
from fastapi.responses import Response

from ogx_api.common.upload_limits import (
    PreReadUploadFile,
    read_upload_with_size_limit,
)
from ogx_api.router_utils import create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1ALPHA

from .api import Skills
from .models import (
    MAX_ZIP_SIZE_BYTES,
    ListSkillsRequest,
    ListSkillsResponse,
    ListSkillVersionsRequest,
    ListSkillVersionsResponse,
    Skill,
    SkillDeleteResponse,
    SkillUpdateRequest,
    SkillVersion,
    SkillVersionCreateRequest,
    SkillVersionDeleteResponse,
)

get_list_skills_request = create_query_dependency(ListSkillsRequest)
get_list_skill_versions_request = create_query_dependency(ListSkillVersionsRequest)


def create_router(impl: Skills) -> APIRouter:
    router = APIRouter(
        prefix=f"/{OGX_API_V1ALPHA}",
        tags=["Skills"],
        responses=standard_responses,
    )

    @router.post(
        "/skills",
        response_model=Skill,
        summary="Create skill",
        description="Create a skill by uploading a zip bundle containing a SKILL.md manifest.",
    )
    async def create_skill(
        file: Annotated[UploadFile, File(description="Zip archive containing the skill bundle.")],
    ) -> Skill:
        content = await read_upload_with_size_limit(file, MAX_ZIP_SIZE_BYTES)
        safe_file = PreReadUploadFile(content, filename=file.filename, content_type=file.content_type)
        return await impl.create_skill(safe_file)

    @router.get(
        "/skills",
        response_model=ListSkillsResponse,
        summary="List skills",
        description="List all skills.",
    )
    async def list_skills(
        request: Annotated[ListSkillsRequest, Depends(get_list_skills_request)],
    ) -> ListSkillsResponse:
        return await impl.list_skills(request)

    @router.get(
        "/skills/{skill_id}",
        response_model=Skill,
        summary="Get skill",
        description="Get metadata for a specific skill.",
    )
    async def get_skill(skill_id: str) -> Skill:
        return await impl.get_skill(skill_id)

    @router.post(
        "/skills/{skill_id}",
        response_model=Skill,
        summary="Update skill",
        description="Update a skill's default version.",
    )
    async def update_skill(skill_id: str, request: SkillUpdateRequest) -> Skill:
        return await impl.update_skill(skill_id, request)

    @router.delete(
        "/skills/{skill_id}",
        response_model=SkillDeleteResponse,
        summary="Delete skill",
        description="Delete a skill and all its versions.",
    )
    async def delete_skill(skill_id: str) -> SkillDeleteResponse:
        return await impl.delete_skill(skill_id)

    @router.get(
        "/skills/{skill_id}/content",
        summary="Get skill content",
        description="Download the default version's zip bundle.",
        responses={
            200: {
                "description": "The skill bundle as a zip archive.",
                "content": {"application/zip": {}},
            },
        },
    )
    async def get_skill_content(skill_id: str) -> Response:
        return await impl.get_skill_content(skill_id)

    @router.post(
        "/skills/{skill_id}/versions",
        response_model=SkillVersion,
        summary="Create skill version",
        description="Upload a new version of a skill.",
    )
    async def create_skill_version(
        skill_id: str,
        file: Annotated[UploadFile, File(description="Zip archive containing the skill bundle.")],
        default: Annotated[bool, Form(description="Whether to set this version as the default.")] = False,
    ) -> SkillVersion:
        content = await read_upload_with_size_limit(file, MAX_ZIP_SIZE_BYTES)
        safe_file = PreReadUploadFile(content, filename=file.filename, content_type=file.content_type)
        request = SkillVersionCreateRequest(default=default)
        return await impl.create_skill_version(skill_id, request, safe_file)

    @router.get(
        "/skills/{skill_id}/versions",
        response_model=ListSkillVersionsResponse,
        summary="List skill versions",
        description="List all versions of a skill.",
    )
    async def list_skill_versions(
        skill_id: str,
        request: Annotated[ListSkillVersionsRequest, Depends(get_list_skill_versions_request)],
    ) -> ListSkillVersionsResponse:
        return await impl.list_skill_versions(skill_id, request)

    @router.get(
        "/skills/{skill_id}/versions/{version}",
        response_model=SkillVersion,
        summary="Get skill version",
        description="Get metadata for a specific skill version.",
    )
    async def get_skill_version(skill_id: str, version: str) -> SkillVersion:
        return await impl.get_skill_version(skill_id, version)

    @router.get(
        "/skills/{skill_id}/versions/{version}/content",
        summary="Get skill version content",
        description="Download a specific version's zip bundle.",
        responses={
            200: {
                "description": "The skill bundle as a zip archive.",
                "content": {"application/zip": {}},
            },
        },
    )
    async def get_skill_version_content(skill_id: str, version: str) -> Response:
        return await impl.get_skill_version_content(skill_id, version)

    @router.delete(
        "/skills/{skill_id}/versions/{version}",
        response_model=SkillVersionDeleteResponse,
        summary="Delete skill version",
        description="Delete a specific version of a skill.",
    )
    async def delete_skill_version(skill_id: str, version: str) -> SkillVersionDeleteResponse:
        return await impl.delete_skill_version(skill_id, version)

    return router
