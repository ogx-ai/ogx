# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Skills API requests and responses.

This module defines the request and response models for the Skills API,
conforming to the OpenAI Skills API for managing versioned skill bundles.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from ogx_api.schema_utils import json_schema_type

MAX_ZIP_SIZE_BYTES = 50 * 1024 * 1024
MAX_UNCOMPRESSED_FILE_SIZE_BYTES = 25 * 1024 * 1024
MAX_FILES_PER_VERSION = 500


@json_schema_type
class SkillFile(BaseModel):
    """A file within a skill bundle."""

    name: str = Field(description="Relative path within the bundle (e.g., 'analyze.py')")
    size: int = Field(description="File size in bytes")
    content_type: str = Field(description="MIME type of the file")


@json_schema_type
class SkillVersion(BaseModel):
    """A specific version of a skill."""

    version: str = Field(description="Version number as a string")
    skill_id: str = Field(description="ID of the parent skill")
    file_manifest: list[SkillFile] = Field(default_factory=list, description="List of files in this version")
    created_at: int = Field(description="Unix timestamp when this version was created")


@json_schema_type
class Skill(BaseModel):
    """A skill resource representing a versioned bundle of code and data.

    Skills are zip archives containing a SKILL.md manifest and supporting files.
    They are mounted into container filesystems at /mnt/skills/{skill_name}/.
    """

    id: str = Field(description="Unique identifier for the skill")
    object: Literal["skill"] = "skill"
    name: str = Field(description="Human-readable name from SKILL.md frontmatter or create request")
    description: str | None = Field(default=None, description="Description of what the skill does")
    default_version: str = Field(default="1", description="Version used when no version is specified")
    latest_version: str = Field(default="1", description="Most recently uploaded version number")
    created_at: int = Field(description="Unix timestamp when the skill was created")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional key-value metadata")


@json_schema_type
class SkillCreateRequest(BaseModel):
    """Request to create a new skill. Bundle content is provided via multipart form data."""

    name: str | None = Field(default=None, description="Skill name (overrides SKILL.md frontmatter)")
    description: str | None = Field(default=None, description="Skill description (overrides SKILL.md frontmatter)")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional key-value metadata")


@json_schema_type
class SkillUpdateRequest(BaseModel):
    """Request to update a skill's default version."""

    default_version: str = Field(description="Version number to set as the default")


@json_schema_type
class SkillDeleteResponse(BaseModel):
    """Response from deleting a skill."""

    id: str = Field(description="ID of the deleted skill")
    object: Literal["skill"] = "skill"
    deleted: bool = Field(default=True, description="Whether the skill was successfully deleted")


@json_schema_type
class SkillVersionDeleteResponse(BaseModel):
    """Response from deleting a skill version."""

    id: str = Field(description="ID of the deleted skill")
    version: str = Field(description="Version that was deleted")
    object: Literal["skill.version"] = "skill.version"
    deleted: bool = Field(default=True, description="Whether the version was successfully deleted")


@json_schema_type
class ListSkillsRequest(BaseModel):
    """Request parameters for listing skills."""

    after: str | None = Field(default=None, description="Cursor for pagination")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    order: Literal["asc", "desc"] = Field(default="desc", description="Sort order by created_at")


@json_schema_type
class ListSkillsResponse(BaseModel):
    """Response from listing skills."""

    object: Literal["list"] = "list"
    data: list[Skill] = Field(description="List of skill objects")
    has_more: bool = Field(default=False, description="Whether there are more results")
    first_id: str | None = Field(default=None, description="ID of the first item in the list")
    last_id: str | None = Field(default=None, description="ID of the last item in the list")


@json_schema_type
class ListSkillVersionsRequest(BaseModel):
    """Request parameters for listing skill versions."""

    after: str | None = Field(default=None, description="Cursor for pagination")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    order: Literal["asc", "desc"] = Field(default="desc", description="Sort order by version")


@json_schema_type
class ListSkillVersionsResponse(BaseModel):
    """Response from listing skill versions."""

    object: Literal["list"] = "list"
    data: list[SkillVersion] = Field(description="List of skill version objects")
    has_more: bool = Field(default=False, description="Whether there are more results")
    first_id: str | None = Field(default=None, description="ID of the first item in the list")
    last_id: str | None = Field(default=None, description="ID of the last item in the list")


class SkillManifest(BaseModel):
    """Parsed content of a SKILL.md manifest file. Internal type, not exposed in the API."""

    name: str | None = None
    description: str | None = None
    version: str | None = None
    tools: list[dict[str, Any]] | None = None
    instructions: str = ""


class SkillBundle(BaseModel):
    """Internal representation of a skill bundle for mounting into containers."""

    skill_id: str
    skill_name: str
    version: str
    file_id: str
    manifest: SkillManifest
