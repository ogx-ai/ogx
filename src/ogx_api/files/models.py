# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ogx_api.common.responses import Order
from ogx_api.schema_utils import json_schema_type


class OpenAIFilePurpose(StrEnum):
    """
    Valid purpose values for OpenAI Files API.
    """

    ASSISTANTS = "assistants"
    BATCH = "batch"


@json_schema_type
class OpenAIFileObject(BaseModel):
    """OpenAI File object as defined in the OpenAI Files API."""

    object: Literal["file"] = Field(default="file", description="The object type, which is always 'file'.")
    id: str = Field(..., description="The file identifier, which can be referenced in the API endpoints.")
    bytes: int = Field(..., description="The size of the file, in bytes.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the file was created.")
    expires_at: int | None = Field(
        default=None, description="The Unix timestamp (in seconds) for when the file expires."
    )
    filename: str = Field(..., description="The name of the file.")
    purpose: OpenAIFilePurpose = Field(..., description="The intended purpose of the file.")
    status: Literal["uploaded", "processed", "error"] = Field(
        ...,
        description="Deprecated. The current status of the file.",
        deprecated=True,
    )
    status_details: str = Field(
        ...,
        description="Deprecated. For details on why a fine-tuning training file failed validation, see the error field on fine_tuning.job.",
        deprecated=True,
    )


@json_schema_type
class ExpiresAfter(BaseModel):
    """Control expiration of uploaded files."""

    MIN: ClassVar[int] = 3600  # 1 hour
    MAX: ClassVar[int] = 2592000  # 30 days

    anchor: Literal["created_at"] = Field(..., description="The anchor point for expiration, must be 'created_at'.")
    seconds: int = Field(
        ..., ge=MIN, le=MAX, description="Seconds until expiration, between 3600 (1 hour) and 2592000 (30 days)."
    )


@json_schema_type
class ListOpenAIFileResponse(BaseModel):
    """Response for listing files in OpenAI Files API."""

    data: list[OpenAIFileObject] = Field(..., description="The list of files.")
    has_more: bool = Field(..., description="Whether there are more files available beyond this page.")
    first_id: str = Field(..., description="The ID of the first file in the list for pagination.")
    last_id: str = Field(..., description="The ID of the last file in the list for pagination.")
    object: Literal["list"] = Field(default="list", description="The object type, which is always 'list'.")


@json_schema_type
class OpenAIFileDeleteResponse(BaseModel):
    """Response for deleting a file in OpenAI Files API."""

    id: str = Field(..., description="The file identifier that was deleted.")
    object: Literal["file"] = Field(default="file", description="The object type, which is always 'file'.")
    deleted: bool = Field(..., description="Whether the file was successfully deleted.")


@json_schema_type
class ListFilesRequest(BaseModel):
    """Request model for listing files."""

    after: str | None = Field(default=None, description="A cursor for pagination. Returns files after this ID.")
    limit: int | None = Field(default=10000, description="Maximum number of files to return (1-10,000).")
    order: Order | None = Field(default=Order.desc, description="Sort order by created_at timestamp ('asc' or 'desc').")
    purpose: OpenAIFilePurpose | None = Field(default=None, description="Filter files by purpose.")


@json_schema_type
class RetrieveFileRequest(BaseModel):
    """Request model for retrieving a file."""

    file_id: str = Field(..., description="The ID of the file to retrieve.")


@json_schema_type
class DeleteFileRequest(BaseModel):
    """Request model for deleting a file."""

    file_id: str = Field(..., description="The ID of the file to delete.")


@json_schema_type
class RetrieveFileContentRequest(BaseModel):
    """Request model for retrieving file content."""

    file_id: str = Field(..., description="The ID of the file to retrieve content from.")


@json_schema_type
class UploadFileRequest(BaseModel):
    """Request model for uploading a file."""

    purpose: OpenAIFilePurpose = Field(..., description="The intended purpose of the uploaded file.")
    expires_after: ExpiresAfter | None = Field(default=None, description="Optional expiration settings for the file.")
