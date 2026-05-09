# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""VectorIO API protocol definition.

This module contains the VectorIO protocol definition.
Pydantic models are defined in ogx_api.vector_io.models.
The FastAPI router is defined in ogx_api.vector_io.fastapi_routes.
"""

from typing import Annotated, Protocol, runtime_checkable

from fastapi import Body, Query

from ogx_api.vector_stores import VectorStore

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


class VectorStoreTable(Protocol):
    """Protocol for looking up vector store definitions."""

    def get_vector_store(self, vector_store_id: str) -> VectorStore | None: ...


@runtime_checkable
class VectorIO(Protocol):
    """Protocol for vector database operations and OpenAI-compatible vector store management."""

    vector_store_table: VectorStoreTable | None = None

    # this will just block now until chunks are inserted, but it should
    # probably return a Job instance which can be polled for completion
    async def insert_chunks(
        self,
        request: InsertChunksRequest,
    ) -> None:
        """Insert embedded chunks into a vector database."""
        ...

    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        """Query chunks from a vector database."""
        ...

    # OpenAI Vector Stores API endpoints
    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        """Creates a vector store.

        Generate an OpenAI-compatible vector store with the given parameters.
        """
        ...

    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        """Returns a list of vector stores."""
        ...

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store."""
        ...

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIUpdateVectorStoreRequest,
    ) -> VectorStoreObject:
        """Updates a vector store."""
        ...

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store."""
        ...

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        request: OpenAISearchVectorStoreRequest,
    ) -> VectorStoreSearchResponsePage:
        """Search for chunks in a vector store.

        Searches a vector store for relevant chunks based on a query and optional file attribute filters.
        """
        ...

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIAttachFileRequest,
    ) -> VectorStoreFileObject:
        """Attach a file to a vector store."""
        ...

    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        """List files in a vector store."""
        ...

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        """Retrieves a vector store file."""
        ...

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
        include_embeddings: Annotated[bool | None, Query()] = False,
        include_metadata: Annotated[bool | None, Query()] = False,
    ) -> VectorStoreFileContentResponse:
        """Retrieves the contents of a vector store file."""
        ...

    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        request: OpenAIUpdateVectorStoreFileRequest,
    ) -> VectorStoreFileObject:
        """Updates a vector store file."""
        ...

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        """Delete a vector store file."""
        ...

    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        """Create a vector store file batch.

        Generate an OpenAI-compatible vector store file batch for the given vector store.
        """
        ...

    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Retrieve a vector store file batch."""
        ...

    async def openai_list_files_in_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
        after: str | None = None,
        before: str | None = None,
        filter: str | None = None,
        limit: int | None = 20,
        order: str | None = "desc",
    ) -> VectorStoreFilesListInBatchResponse:
        """Returns a list of vector store files in a batch."""
        ...

    async def openai_cancel_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Cancels a vector store file batch."""
        ...


__all__ = [
    "VectorIO",
    "VectorStoreTable",
]
