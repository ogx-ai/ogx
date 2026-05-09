# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the VectorIO API.

This module defines the FastAPI router for the VectorIO API using standard
FastAPI route decorators.
"""

from __future__ import annotations

from typing import Annotated, NoReturn, cast

from fastapi import APIRouter, Body, HTTPException, Path, Query, Request, Response, status

from ogx_api.router_utils import standard_responses
from ogx_api.version import OGX_API_V1

from .api import VectorIO
from .models import (
    InsertChunksRequest,
    OpenAIAttachFileRequest,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    OpenAISearchVectorStoreRequest,
    OpenAIUpdateVectorStoreFileRequest,
    OpenAIUpdateVectorStoreRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)


def create_router(impl: VectorIO) -> APIRouter:
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["VectorIO"],
        responses=standard_responses,
    )

    @router.post(
        "/vector-io/insert",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        summary="Insert embedded chunks into a vector database.",
        description="Insert embedded chunks into a vector database.",
        responses={204: {"description": "Chunks were inserted."}},
    )
    async def insert_chunks(request: Annotated[InsertChunksRequest, Body(...)]) -> None:
        await impl.insert_chunks(request)
        return None

    @router.post(
        "/vector-io/query",
        response_model=QueryChunksResponse,
        summary="Query chunks from a vector database.",
        description="Query chunks from a vector database.",
        responses={200: {"description": "A QueryChunksResponse."}},
    )
    async def query_chunks(request: Annotated[QueryChunksRequest, Body(...)]) -> QueryChunksResponse:
        return await impl.query_chunks(request)

    @router.post(
        "/vector_stores",
        response_model=VectorStoreObject,
        summary="Create a vector store (OpenAI-compatible).",
        description="Create a vector store (OpenAI-compatible).",
        responses={200: {"description": "The created vector store."}},
    )
    async def openai_create_vector_store(
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        return await impl.openai_create_vector_store(params)

    @router.get(
        "/vector_stores",
        response_model=VectorStoreListResponse,
        summary="List vector stores (OpenAI-compatible).",
        description="List vector stores (OpenAI-compatible).",
        responses={200: {"description": "A list of vector stores."}},
    )
    async def openai_list_vector_stores(
        limit: Annotated[
            int | None, Query(ge=1, le=100, description="Maximum number of vector stores to return.")
        ] = 20,
        order: Annotated[str | None, Query(description="Sort order by created_at: asc or desc.")] = "desc",
        after: Annotated[
            str | None,
            Query(
                description="Pagination cursor (after).",
            ),
        ] = None,
        before: Annotated[
            str | None,
            Query(
                description="Pagination cursor (before).",
            ),
        ] = None,
    ) -> VectorStoreListResponse:
        return await impl.openai_list_vector_stores(limit=limit, order=order, after=after, before=before)

    @router.get(
        "/vector_stores/{vector_store_id}",
        response_model=VectorStoreObject,
        summary="Retrieve a vector store (OpenAI-compatible).",
        description="Retrieve a vector store (OpenAI-compatible).",
        responses={200: {"description": "The vector store."}},
    )
    async def openai_retrieve_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
    ) -> VectorStoreObject:
        return await impl.openai_retrieve_vector_store(vector_store_id=vector_store_id)

    @router.post(
        "/vector_stores/{vector_store_id}",
        response_model=VectorStoreObject,
        summary="Update a vector store (OpenAI-compatible).",
        description="Update a vector store (OpenAI-compatible).",
        responses={200: {"description": "The updated vector store."}},
    )
    async def openai_update_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        request: Annotated[OpenAIUpdateVectorStoreRequest, Body(...)],
    ) -> VectorStoreObject:
        return await impl.openai_update_vector_store(
            vector_store_id=vector_store_id,
            request=request,
        )

    @router.delete(
        "/vector_stores/{vector_store_id}",
        response_model=VectorStoreDeleteResponse,
        summary="Delete a vector store (OpenAI-compatible).",
        description="Delete a vector store (OpenAI-compatible).",
        responses={200: {"description": "Vector store deleted."}},
    )
    async def openai_delete_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
    ) -> VectorStoreDeleteResponse:
        return await impl.openai_delete_vector_store(vector_store_id=vector_store_id)

    @router.post(
        "/vector_stores/{vector_store_id}/search",
        response_model=VectorStoreSearchResponsePage,
        summary="Search a vector store (OpenAI-compatible).",
        description="Search a vector store (OpenAI-compatible).",
        responses={200: {"description": "Search results."}},
    )
    async def openai_search_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        request: Annotated[OpenAISearchVectorStoreRequest, Body(...)],
    ) -> VectorStoreSearchResponsePage:
        return await impl.openai_search_vector_store(
            vector_store_id=vector_store_id,
            request=request,
        )

    @router.post(
        "/vector_stores/{vector_store_id}/files",
        response_model=VectorStoreFileObject,
        summary="Attach a file to a vector store (OpenAI-compatible).",
        description="Attach a file to a vector store (OpenAI-compatible).",
        responses={200: {"description": "The attached file."}},
    )
    async def openai_attach_file_to_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        request: Annotated[OpenAIAttachFileRequest, Body(...)],
    ) -> VectorStoreFileObject:
        return await impl.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            request=request,
        )

    @router.get(
        "/vector_stores/{vector_store_id}/files",
        response_model=VectorStoreListFilesResponse,
        summary="List files in a vector store (OpenAI-compatible).",
        description="List files in a vector store (OpenAI-compatible).",
        responses={200: {"description": "A list of files in the vector store."}},
    )
    async def openai_list_files_in_vector_store(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        limit: Annotated[int | None, Query(ge=1, le=100, description="Maximum number of files to return.")] = 20,
        order: Annotated[str | None, Query(description="Sort order by created_at: asc or desc.")] = "desc",
        after: Annotated[
            str | None,
            Query(
                description="Pagination cursor (after).",
            ),
        ] = None,
        before: Annotated[
            str | None,
            Query(
                description="Pagination cursor (before).",
            ),
        ] = None,
        filter: Annotated[VectorStoreFileStatus | None, Query(description="Filter by file status.")] = None,
    ) -> VectorStoreListFilesResponse:
        return await impl.openai_list_files_in_vector_store(
            vector_store_id=vector_store_id,
            limit=limit,
            order=order,
            after=after,
            before=before,
            filter=filter,
        )

    @router.get(
        "/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=VectorStoreFileObject,
        summary="Retrieve a vector store file (OpenAI-compatible).",
        description="Retrieve a vector store file (OpenAI-compatible).",
        responses={200: {"description": "The vector store file."}},
    )
    async def openai_retrieve_vector_store_file(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
    ) -> VectorStoreFileObject:
        return await impl.openai_retrieve_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)

    @router.get(
        "/vector_stores/{vector_store_id}/files/{file_id}/content",
        response_model=VectorStoreFileContentResponse,
        summary="Retrieve vector store file contents (OpenAI-compatible).",
        description="Retrieve vector store file contents (OpenAI-compatible).",
        responses={200: {"description": "The vector store file contents."}},
    )
    async def openai_retrieve_vector_store_file_contents(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
        include_embeddings: Annotated[bool | None, Query(description="Include embedding vectors.")] = False,
        include_metadata: Annotated[bool | None, Query(description="Include chunk metadata.")] = False,
    ) -> VectorStoreFileContentResponse:
        return await impl.openai_retrieve_vector_store_file_contents(
            vector_store_id=vector_store_id,
            file_id=file_id,
            include_embeddings=include_embeddings,
            include_metadata=include_metadata,
        )

    @router.post(
        "/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=VectorStoreFileObject,
        summary="Update a vector store file (OpenAI-compatible).",
        description="Update a vector store file (OpenAI-compatible).",
        responses={200: {"description": "The updated vector store file."}},
    )
    async def openai_update_vector_store_file(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
        request: Annotated[OpenAIUpdateVectorStoreFileRequest, Body(...)],
    ) -> VectorStoreFileObject:
        return await impl.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=request,
        )

    @router.delete(
        "/vector_stores/{vector_store_id}/files/{file_id}",
        response_model=VectorStoreFileDeleteResponse,
        summary="Delete a vector store file (OpenAI-compatible).",
        description="Delete a vector store file (OpenAI-compatible).",
        responses={200: {"description": "The vector store file was deleted."}},
    )
    async def openai_delete_vector_store_file(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        file_id: Annotated[str, Path(description="The file identifier.")],
    ) -> VectorStoreFileDeleteResponse:
        return await impl.openai_delete_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)

    @router.post(
        "/vector_stores/{vector_store_id}/file_batches",
        response_model=VectorStoreFileBatchObject,
        summary="Create a vector store file batch (OpenAI-compatible).",
        description="Create a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "The created file batch."}},
    )
    async def openai_create_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
        request: Request = _MISSING_REQUEST,
    ) -> VectorStoreFileBatchObject:
        try:
            return await impl.openai_create_vector_store_file_batch(vector_store_id=vector_store_id, params=params)
        except ValueError as exc:
            _raise_or_http_400_for_value_error(request, exc)

    @router.get(
        "/vector_stores/{vector_store_id}/file_batches/{batch_id}",
        response_model=VectorStoreFileBatchObject,
        summary="Retrieve a vector store file batch (OpenAI-compatible).",
        description="Retrieve a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "The file batch."}},
    )
    async def openai_retrieve_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        batch_id: Annotated[str, Path(description="The file batch identifier.")],
        request: Request = _MISSING_REQUEST,
    ) -> VectorStoreFileBatchObject:
        try:
            return await impl.openai_retrieve_vector_store_file_batch(
                batch_id=batch_id, vector_store_id=vector_store_id
            )
        except ValueError as exc:
            _raise_or_http_400_for_value_error(request, exc)

    @router.get(
        "/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
        response_model=VectorStoreFilesListInBatchResponse,
        summary="List files in a vector store file batch (OpenAI-compatible).",
        description="List files in a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "A list of files in the file batch."}},
    )
    async def openai_list_files_in_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        batch_id: Annotated[str, Path(description="The file batch identifier.")],
        after: Annotated[
            str | None,
            Query(
                description="Pagination cursor (after).",
            ),
        ] = None,
        before: Annotated[
            str | None,
            Query(
                description="Pagination cursor (before).",
            ),
        ] = None,
        filter: Annotated[str | None, Query(description="Filter by file status.")] = None,
        limit: Annotated[int | None, Query(ge=1, le=100, description="Maximum number of files to return.")] = 20,
        order: Annotated[str | None, Query(description="Sort order by created_at: asc or desc.")] = "desc",
    ) -> VectorStoreFilesListInBatchResponse:
        return await impl.openai_list_files_in_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
            after=after,
            before=before,
            filter=filter,
            limit=limit,
            order=order,
        )

    @router.post(
        "/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
        response_model=VectorStoreFileBatchObject,
        summary="Cancel a vector store file batch (OpenAI-compatible).",
        description="Cancel a vector store file batch (OpenAI-compatible).",
        responses={200: {"description": "The cancelled file batch."}},
    )
    async def openai_cancel_vector_store_file_batch(
        vector_store_id: Annotated[str, Path(description="The vector store identifier.")],
        batch_id: Annotated[str, Path(description="The file batch identifier.")],
    ) -> VectorStoreFileBatchObject:
        return await impl.openai_cancel_vector_store_file_batch(batch_id=batch_id, vector_store_id=vector_store_id)

    return router


_MISSING_REQUEST: Request = cast(Request, None)


def _raise_or_http_400_for_value_error(request: Request, exc: ValueError) -> NoReturn:
    # In library mode, FastAPI doesn't inject a Request.
    if request is _MISSING_REQUEST:
        raise
    raise HTTPException(status_code=400, detail=str(exc)) from exc
