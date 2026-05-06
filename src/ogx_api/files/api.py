# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from fastapi import Response, UploadFile

from .models import (
    DeleteFileRequest,
    ListFilesRequest,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)


@runtime_checkable
class Files(Protocol):
    """Files API for managing file uploads and retrieval."""

    async def openai_upload_file(
        self,
        request: UploadFileRequest,
        file: UploadFile,
    ) -> OpenAIFileObject: ...

    async def openai_list_files(
        self,
        request: ListFilesRequest,
    ) -> ListOpenAIFileResponse: ...

    async def openai_retrieve_file(
        self,
        request: RetrieveFileRequest,
    ) -> OpenAIFileObject: ...

    async def openai_delete_file(
        self,
        request: DeleteFileRequest,
    ) -> OpenAIFileDeleteResponse: ...

    async def openai_retrieve_file_content(
        self,
        request: RetrieveFileContentRequest,
    ) -> Response: ...
