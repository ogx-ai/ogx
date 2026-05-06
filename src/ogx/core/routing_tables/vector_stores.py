# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import (
    VectorStoreWithOwner,
)
from ogx.log import get_logger

# Removed VectorStores import to avoid exposing public API
from ogx_api import (
    InsertChunksRequest,
    ModelNotFoundError,
    ModelType,
    ModelTypeError,
    OpenAIAttachFileRequest,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAISearchVectorStoreRequest,
    OpenAIUpdateVectorStoreFileRequest,
    OpenAIUpdateVectorStoreRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    ResourceType,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)

from .common import CommonRoutingTableImpl, lookup_model

logger = get_logger(name=__name__, category="core::routing_tables")


class VectorStoresRoutingTable(CommonRoutingTableImpl):
    """Internal routing table for vector_store operations.

    Does not inherit from VectorStores to avoid exposing public API endpoints.
    Only provides internal routing functionality for VectorIORouter.
    """

    def __init__(
        self,
        impls_by_provider_id: dict[str, Any],
        dist_registry: Any,
        policy: list[Any],
    ) -> None:
        super().__init__(impls_by_provider_id, dist_registry, policy)
        self.vector_io_router = None  # Will be set post-instantiation

    # Internal methods only - no public API exposure

    async def list_vector_stores(self) -> list[VectorStoreWithOwner]:
        """List all registered vector stores."""
        return await self.get_all_with_type(ResourceType.vector_store.value)

    async def register_vector_store(
        self,
        vector_store_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_store_id: str | None = None,
        vector_store_name: str | None = None,
    ) -> Any:
        if provider_id is None:
            if len(self.impls_by_provider_id) > 0:
                provider_id = list(self.impls_by_provider_id.keys())[0]
                if len(self.impls_by_provider_id) > 1:
                    logger.warning(
                        "No provider specified and multiple providers available. Arbitrarily selected the first provider .",
                        provider_id=provider_id,
                    )
            else:
                raise ValueError("No provider available. Please configure a vector_io provider.")
        model = await lookup_model(self, embedding_model)
        if model is None:
            raise ModelNotFoundError(embedding_model)
        if model.model_type != ModelType.embedding:
            raise ModelTypeError(embedding_model, model.model_type, ModelType.embedding)

        vector_store = VectorStoreWithOwner(
            identifier=vector_store_id,
            type=ResourceType.vector_store.value,
            provider_id=provider_id,
            provider_resource_id=provider_vector_store_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            vector_store_name=vector_store_name,
        )
        await self.register_object(vector_store)
        return vector_store

    async def insert_chunks(
        self,
        request: InsertChunksRequest,
    ) -> None:
        await self.assert_action_allowed("update", "vector_store", request.vector_store_id)
        provider = await self.get_provider_impl(request.vector_store_id)
        return await provider.insert_chunks(request)

    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        await self.assert_action_allowed("read", "vector_store", request.vector_store_id)
        provider = await self.get_provider_impl(request.vector_store_id)
        return await provider.query_chunks(request)

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        await self.assert_action_allowed("read", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store(vector_store_id)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIUpdateVectorStoreRequest,
    ) -> VectorStoreObject:
        await self.assert_action_allowed("update", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_update_vector_store(vector_store_id, request)

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        await self.assert_action_allowed("delete", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        result = await provider.openai_delete_vector_store(vector_store_id)
        await self.unregister_vector_store(vector_store_id)
        return result

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Remove the vector store from the routing table registry."""
        try:
            vector_store_obj = await self.get_object_by_identifier("vector_store", vector_store_id)
            if vector_store_obj:
                await self.unregister_object(vector_store_obj)
        except Exception as e:
            # Log the error but don't fail the operation
            logger.warning(
                "Failed to unregister vector store from routing table", vector_store_id=vector_store_id, error=str(e)
            )

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        request: OpenAISearchVectorStoreRequest,
    ) -> VectorStoreSearchResponsePage:
        await self.assert_action_allowed("read", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_search_vector_store(vector_store_id, request)

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIAttachFileRequest,
    ) -> VectorStoreFileObject:
        await self.assert_action_allowed("update", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_attach_file_to_vector_store(vector_store_id, request)

    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        await self.assert_action_allowed("read", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_list_files_in_vector_store(
            vector_store_id=vector_store_id,
            limit=limit,
            order=order,
            after=after,
            before=before,
            filter=filter,
        )

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        await self.assert_action_allowed("read", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
        include_embeddings: bool | None = False,
        include_metadata: bool | None = False,
    ) -> VectorStoreFileContentResponse:
        await self.assert_action_allowed("read", "vector_store", vector_store_id)

        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file_contents(
            vector_store_id=vector_store_id,
            file_id=file_id,
            include_embeddings=include_embeddings,
            include_metadata=include_metadata,
        )

    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        request: OpenAIUpdateVectorStoreFileRequest,
    ) -> VectorStoreFileObject:
        await self.assert_action_allowed("update", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=request,
        )

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        await self.assert_action_allowed("delete", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_delete_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )

    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    ) -> VectorStoreFileBatchObject:
        await self.assert_action_allowed("update", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_create_vector_store_file_batch(
            vector_store_id=vector_store_id,
            params=params,
        )

    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        await self.assert_action_allowed("read", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_retrieve_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
        )

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
        await self.assert_action_allowed("read", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_list_files_in_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
            after=after,
            before=before,
            filter=filter,
            limit=limit,
            order=order,
        )

    async def openai_cancel_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        await self.assert_action_allowed("update", "vector_store", vector_store_id)
        provider = await self.get_provider_impl(vector_store_id)
        return await provider.openai_cancel_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
        )
