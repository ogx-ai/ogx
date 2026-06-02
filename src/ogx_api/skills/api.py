# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from fastapi import Response, UploadFile

from .models import (
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

__all__ = ["Skills"]


@runtime_checkable
class Skills(Protocol):
    """Skills API for managing versioned skill bundles.

    Skills are zip archives containing a SKILL.md manifest and supporting files.
    Conforms to the OpenAI Skills API wire format.
    """

    async def create_skill(
        self,
        file: UploadFile,
    ) -> Skill: ...

    async def list_skills(
        self,
        request: ListSkillsRequest,
    ) -> ListSkillsResponse: ...

    async def get_skill(self, skill_id: str) -> Skill: ...

    async def update_skill(
        self,
        skill_id: str,
        request: SkillUpdateRequest,
    ) -> Skill: ...

    async def delete_skill(self, skill_id: str) -> SkillDeleteResponse: ...

    async def get_skill_content(self, skill_id: str) -> Response: ...

    async def create_skill_version(
        self,
        skill_id: str,
        request: SkillVersionCreateRequest,
        file: UploadFile,
    ) -> SkillVersion: ...

    async def list_skill_versions(
        self,
        skill_id: str,
        request: ListSkillVersionsRequest,
    ) -> ListSkillVersionsResponse: ...

    async def get_skill_version(
        self,
        skill_id: str,
        version: str,
    ) -> SkillVersion: ...

    async def get_skill_version_content(
        self,
        skill_id: str,
        version: str,
    ) -> Response: ...

    async def delete_skill_version(
        self,
        skill_id: str,
        version: str,
    ) -> SkillVersionDeleteResponse: ...
