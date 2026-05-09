# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
import uuid
from typing import Annotated

from fastapi import Body

from ogx.core.datatypes import VectorStoresConfig
from ogx.log import get_logger
from ogx.providers.utils.vector_io.filters import parse_filter
from ogx.telemetry.vector_io_metrics import (
    create_vector_metric_attributes,
    vector_chunks_processed_total,
    vector_deletes_total,
    vector_files_total,
    vector_insert_duration,
    vector_inserts_total,
    vector_queries_total,
    vector_retrieval_duration,
    vector_stores_total,
)
from ogx_api import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    HealthResponse,
    HealthStatus,
    Inference,
    InsertChunksRequest,
    ModelNotFoundError,
    ModelType,
    ModelTypeError,
    OpenAIAttachFileRequest,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    OpenAISearchVectorStoreRequest,
    OpenAIUpdateVectorStoreFileRequest,
    OpenAIUpdateVectorStoreRequest,
    OpenAIUserMessageParam,
    QueryChunksRequest,
    QueryChunksResponse,
    RoutingTable,
    VectorIO,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
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

logger = get_logger(name=__name__, category="core::routers")


class VectorIORouter(VectorIO):
    """Routes to an provider based on the vector db identifier"""

    def __init__(
        self,
        routing_table: RoutingTable,
        vector_stores_config: VectorStoresConfig | None = None,
        inference_api: Inference | None = None,
    ) -> None:
        self.routing_table = routing_table
        self.vector_stores_config = vector_stores_config
        self.inference_api = inference_api

    async def initialize(self) -> None:
        logger.debug("VectorIORouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("VectorIORouter.shutdown")
        pass

    def _get_provider_id(self, vector_store_id: str) -> str:
        """Get the provider ID for a vector store for metrics labeling (best-effort).

        Uses the same in-memory cache (get_cached) that the routing table's
        get_provider_impl uses when dispatching operations, so this does NOT
        cause an extra DB/async lookup on the hot path.

        Returns "unknown" only as a fallback so that a metrics-label lookup
        failure never blocks the actual operation.
        """
        try:
            obj = self.routing_table.dist_registry.get_cached("vector_store", vector_store_id)
            if obj is None:
                logger.warning("Vector store not found in registry cache", vector_store_id=vector_store_id)
                return "unknown"
            return obj.provider_id
        except Exception:
            logger.warning(
                "Could not resolve provider for vector store", vector_store_id=vector_store_id, exc_info=True
            )
            return "unknown"

    async def _rewrite_query_for_search(self, query: str) -> str:
        """Rewrite a search query using the configured LLM model for better retrieval results."""
        if (
            not self.vector_stores_config
            or not self.vector_stores_config.rewrite_query_params
            or not self.vector_stores_config.rewrite_query_params.model
        ):
            logger.warning(
                "User is trying to use vector_store query rewriting, but it is not configured. Please configure rewrite_query_params.model in vector_stores config."
            )
            raise ValueError("Query rewriting is not available")

        if not self.inference_api:
            logger.warning("Query rewriting requires inference API but it is not available")
            raise ValueError("Query rewriting is not available")

        model = self.vector_stores_config.rewrite_query_params.model
        model_id = f"{model.provider_id}/{model.model_id}"

        prompt = self.vector_stores_config.rewrite_query_params.prompt.format(query=query)

        request = OpenAIChatCompletionRequestWithExtraBody(
            model=model_id,
            messages=[OpenAIUserMessageParam(role="user", content=prompt)],
            max_tokens=self.vector_stores_config.rewrite_query_params.max_tokens or 100,
            temperature=self.vector_stores_config.rewrite_query_params.temperature or 0.3,
        )

        try:
            response = await self.inference_api.openai_chat_completion(request)
            content = response.choices[0].message.content
            if content is None:
                logger.error("LLM returned None content for query rewriting. Model", model_id=model_id)
                raise RuntimeError("Query rewrite failed due to an internal error")
            rewritten_query: str = content.strip()
            return rewritten_query
        except Exception as e:
            logger.error("Query rewrite failed with LLM call error", model_id=model_id, error=str(e))
            raise RuntimeError("Query rewrite failed due to an internal error") from e

    async def _get_embedding_model_dimension(self, embedding_model_id: str) -> int:
        """Get the embedding dimension for a specific embedding model."""
        all_models = await self.routing_table.get_all_with_type("model")

        for model in all_models:
            if model.identifier == embedding_model_id and model.model_type == ModelType.embedding:
                dimension = model.metadata.get("embedding_dimension")
                if dimension is None:
                    raise ValueError(f"Embedding model '{embedding_model_id}' has no embedding_dimension in metadata")
                return int(dimension)

        raise ValueError(f"Embedding model '{embedding_model_id}' not found or not an embedding model")

    async def insert_chunks(
        self,
        request: InsertChunksRequest,
    ) -> None:
        doc_ids = [chunk.document_id for chunk in request.chunks[:3]]
        logger.debug(
            "VectorIORouter.insert_chunks",
            vector_store_id=request.vector_store_id,
            chunk_count=len(request.chunks),
            ttl_seconds=request.ttl_seconds,
            doc_ids=doc_ids,
        )
        start_time = time.perf_counter()
        num_chunks = len(request.chunks)
        provider_id = self._get_provider_id(request.vector_store_id)
        metric_attrs = create_vector_metric_attributes(
            vector_db=request.vector_store_id,
            operation="chunks",
            provider=provider_id,
        )

        try:
            result = await self.routing_table.insert_chunks(request)
            duration = time.perf_counter() - start_time
            success_attrs = {**metric_attrs, "status": "success"}
            vector_inserts_total.add(1, success_attrs)
            vector_insert_duration.record(duration, metric_attrs)
            vector_chunks_processed_total.add(num_chunks, metric_attrs)
            return result
        except asyncio.CancelledError:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_inserts_total.add(1, error_attrs)
            vector_insert_duration.record(duration, metric_attrs)
            raise
        except Exception:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_inserts_total.add(1, error_attrs)
            vector_insert_duration.record(duration, metric_attrs)
            raise

    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        logger.debug("VectorIORouter.query_chunks", vector_store_id=request.vector_store_id)
        start_time = time.perf_counter()
        provider_id = self._get_provider_id(request.vector_store_id)
        metric_attrs = create_vector_metric_attributes(
            vector_db=request.vector_store_id,
            operation="query",
            provider=provider_id,
            search_mode="vector",
        )

        try:
            # Handle the no-filters case early
            if not request.params or "filters" not in request.params:
                result = await self.routing_table.query_chunks(request)
            else:
                # Extract and parse filters from request params
                params_copy = dict(request.params)
                filter_data = params_copy.pop("filters")

                try:
                    parsed_filters = parse_filter(filter_data)
                except ValueError as e:
                    logger.error("Invalid filter data", error=str(e))
                    raise ValueError(f"Invalid filter: {e}") from e

                params_copy["filters"] = parsed_filters
                modified_request = QueryChunksRequest(
                    vector_store_id=request.vector_store_id, query=request.query, params=params_copy
                )
                result = await self.routing_table.query_chunks(modified_request)

            duration = time.perf_counter() - start_time
            success_attrs = {**metric_attrs, "status": "success"}
            vector_queries_total.add(1, success_attrs)
            vector_retrieval_duration.record(duration, metric_attrs)
            return result
        except asyncio.CancelledError:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_queries_total.add(1, error_attrs)
            vector_retrieval_duration.record(duration, metric_attrs)
            raise
        except Exception:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_queries_total.add(1, error_attrs)
            vector_retrieval_duration.record(duration, metric_attrs)
            raise

    # OpenAI Vector Stores API endpoints
    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        # Extract ogx-specific parameters from extra_body or metadata
        extra = params.model_extra or {}
        metadata = params.metadata or {}
        embedding_model = extra.get("embedding_model", metadata.get("embedding_model"))
        embedding_dimension = extra.get("embedding_dimension", metadata.get("embedding_dimension"))
        provider_id = extra.get("provider_id", metadata.get("provider_id"))

        # Use default embedding model if not specified
        if (
            embedding_model is None
            and self.vector_stores_config
            and self.vector_stores_config.default_embedding_model is not None
        ):
            # Construct the full model ID with provider prefix
            embedding_provider_id = self.vector_stores_config.default_embedding_model.provider_id
            model_id = self.vector_stores_config.default_embedding_model.model_id
            embedding_model = f"{embedding_provider_id}/{model_id}"

        if embedding_model is not None and embedding_dimension is None:
            if (
                self.vector_stores_config
                and self.vector_stores_config.default_embedding_model is not None
                and self.vector_stores_config.default_embedding_model.embedding_dimensions
            ):
                embedding_dimension = self.vector_stores_config.default_embedding_model.embedding_dimensions
            else:
                embedding_dimension = await self._get_embedding_model_dimension(embedding_model)
        # Validate that embedding model exists and is of the correct type
        if embedding_model is not None:
            model = await self.routing_table.get_object_by_identifier("model", embedding_model)
            if model is None:
                raise ModelNotFoundError(embedding_model)
            if model.model_type != ModelType.embedding:
                raise ModelTypeError(embedding_model, model.model_type, ModelType.embedding)

        # Auto-select provider if not specified
        if provider_id is None:
            num_providers = len(self.routing_table.impls_by_provider_id)
            if num_providers == 0:
                raise ValueError("No vector_io providers available")
            if num_providers > 1:
                available_providers = list(self.routing_table.impls_by_provider_id.keys())
                # Use default configured provider
                if self.vector_stores_config and self.vector_stores_config.default_provider_id:
                    default_provider = self.vector_stores_config.default_provider_id
                    if default_provider in available_providers:
                        provider_id = default_provider
                        logger.debug("Using configured default vector store provider", provider_id=provider_id)
                    else:
                        raise ValueError(
                            f"Configured default vector store provider '{default_provider}' not found. "
                            f"Available providers: {available_providers}"
                        )
                else:
                    raise ValueError(
                        f"Multiple vector_io providers available. Please specify provider_id in extra_body. "
                        f"Available providers: {available_providers}"
                    )
            else:
                provider_id = list(self.routing_table.impls_by_provider_id.keys())[0]

        vector_store_id = f"vs_{uuid.uuid4()}"
        registered_vector_store = await self.routing_table.register_vector_store(
            vector_store_id=vector_store_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            provider_id=provider_id,
            provider_vector_store_id=vector_store_id,
            vector_store_name=params.name,
        )
        provider = await self.routing_table.get_provider_impl(registered_vector_store.identifier)

        # Build extra fields to pass to provider with registered values
        extra_fields: dict[str, str | int | None] = {
            "provider_vector_store_id": registered_vector_store.provider_resource_id,
            "provider_id": registered_vector_store.provider_id,
        }
        if embedding_model is not None:
            extra_fields["embedding_model"] = embedding_model
        if embedding_dimension is not None:
            extra_fields["embedding_dimension"] = embedding_dimension

        # Rebuild params with merged extra fields (Pydantic v2: model_extra is read-only)
        # We need to dump and revalidate to properly merge extra fields
        existing_extra = params.model_extra or {}
        merged_data = {**params.model_dump(exclude_unset=True), **existing_extra, **extra_fields}
        params = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(merged_data)

        # Set chunking strategy explicitly if not provided
        if params.chunking_strategy is None or params.chunking_strategy.type == "auto":
            # actualize the chunking strategy to static
            params.chunking_strategy = VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig(
                    max_chunk_size_tokens=DEFAULT_CHUNK_SIZE_TOKENS,
                    chunk_overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
                )
            )

        result = await provider.openai_create_vector_store(params)
        vector_stores_total.add(
            1,
            create_vector_metric_attributes(provider=provider_id, operation="create"),
        )
        return result

    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        logger.debug("VectorIORouter.openai_list_vector_stores", limit=limit)
        # Route to default provider for now - could aggregate from all providers in the future
        # call retrieve on each vector dbs to get list of vector stores
        vector_stores = await self.routing_table.get_all_with_type("vector_store")
        all_stores = []
        for vector_store in vector_stores:
            try:
                vector_store_obj = await self.routing_table.openai_retrieve_vector_store(vector_store.identifier)
                all_stores.append(vector_store_obj)
            except Exception as e:
                logger.error("Error retrieving vector store", identifier=vector_store.identifier, error=str(e))
                continue

        # Sort by created_at
        reverse_order = order == "desc"
        all_stores.sort(key=lambda x: x.created_at, reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, store in enumerate(all_stores) if store.id == after), -1)
            if after_index >= 0:
                all_stores = all_stores[after_index + 1 :]

        if before:
            before_index = next(
                (i for i, store in enumerate(all_stores) if store.id == before),
                len(all_stores),
            )
            all_stores = all_stores[:before_index]

        # Apply limit
        limited_stores = all_stores[:limit]

        # Determine pagination info
        has_more = len(all_stores) > limit
        first_id = limited_stores[0].id if limited_stores else ""
        last_id = limited_stores[-1].id if limited_stores else ""

        return VectorStoreListResponse(
            data=limited_stores,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        logger.debug("VectorIORouter.openai_retrieve_vector_store", vector_store_id=vector_store_id)
        return await self.routing_table.openai_retrieve_vector_store(vector_store_id)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIUpdateVectorStoreRequest,
    ) -> VectorStoreObject:
        logger.debug("VectorIORouter.openai_update_vector_store", vector_store_id=vector_store_id)

        # Check if provider_id is being changed (not supported)
        if request.metadata and "provider_id" in request.metadata:
            current_store = await self.routing_table.get_object_by_identifier("vector_store", vector_store_id)
            if current_store and current_store.provider_id != request.metadata["provider_id"]:
                raise ValueError("provider_id cannot be changed after vector store creation")

        return await self.routing_table.openai_update_vector_store(
            vector_store_id=vector_store_id,
            request=request,
        )

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        logger.debug("VectorIORouter.openai_delete_vector_store", vector_store_id=vector_store_id)
        provider_id = self._get_provider_id(vector_store_id)
        metric_attrs = create_vector_metric_attributes(
            vector_db=vector_store_id,
            operation="store",
            provider=provider_id,
        )
        try:
            result = await self.routing_table.openai_delete_vector_store(vector_store_id)
            vector_deletes_total.add(1, {**metric_attrs, "status": "success"})
            return result
        except asyncio.CancelledError:
            vector_deletes_total.add(1, {**metric_attrs, "status": "error"})
            raise
        except Exception:
            vector_deletes_total.add(1, {**metric_attrs, "status": "error"})
            raise

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        request: OpenAISearchVectorStoreRequest,
    ) -> VectorStoreSearchResponsePage:
        logger.debug("VectorIORouter.openai_search_vector_store", vector_store_id=vector_store_id)
        start_time = time.perf_counter()
        provider_id = self._get_provider_id(vector_store_id)
        search_mode = getattr(request, "search_mode", "vector")
        metric_attrs = create_vector_metric_attributes(
            vector_db=vector_store_id,
            operation="search",
            provider=provider_id,
            search_mode=search_mode,
        )

        try:
            # Handle query rewriting at the router level
            search_query = request.query
            if request.rewrite_query:
                if isinstance(request.query, list):
                    original_query = " ".join(request.query)
                else:
                    original_query = request.query
                search_query = await self._rewrite_query_for_search(original_query)

            forward_request = request.model_copy()
            forward_request.query = search_query
            forward_request.rewrite_query = False

            result = await self.routing_table.openai_search_vector_store(
                vector_store_id=vector_store_id,
                request=forward_request,
            )

            duration = time.perf_counter() - start_time
            success_attrs = {**metric_attrs, "status": "success"}
            vector_queries_total.add(1, success_attrs)
            vector_retrieval_duration.record(duration, metric_attrs)
            return result
        except asyncio.CancelledError:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_queries_total.add(1, error_attrs)
            vector_retrieval_duration.record(duration, metric_attrs)
            raise
        except Exception:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_queries_total.add(1, error_attrs)
            vector_retrieval_duration.record(duration, metric_attrs)
            raise

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIAttachFileRequest,
    ) -> VectorStoreFileObject:
        logger.debug(
            "VectorIORouter.openai_attach_file_to_vector_store",
            vector_store_id=vector_store_id,
            file_id=request.file_id,
        )
        start_time = time.perf_counter()
        provider_id = self._get_provider_id(vector_store_id)
        metric_attrs = create_vector_metric_attributes(
            vector_db=vector_store_id,
            operation="attach",
            provider=provider_id,
        )

        # Create a copy to modify chunking strategy if needed
        params = request.model_copy()

        if params.chunking_strategy is None or params.chunking_strategy.type == "auto":
            params.chunking_strategy = VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig(
                    max_chunk_size_tokens=DEFAULT_CHUNK_SIZE_TOKENS,
                    chunk_overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
                )
            )

        try:
            result = await self.routing_table.openai_attach_file_to_vector_store(
                vector_store_id=vector_store_id,
                request=params,
            )
            duration = time.perf_counter() - start_time
            success_attrs = {**metric_attrs, "status": "success"}
            vector_files_total.add(1, success_attrs)
            vector_inserts_total.add(1, success_attrs)
            vector_insert_duration.record(duration, metric_attrs)
            return result
        except asyncio.CancelledError:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_files_total.add(1, error_attrs)
            vector_inserts_total.add(1, error_attrs)
            vector_insert_duration.record(duration, metric_attrs)
            raise
        except Exception:
            duration = time.perf_counter() - start_time
            error_attrs = {**metric_attrs, "status": "error"}
            vector_files_total.add(1, error_attrs)
            vector_inserts_total.add(1, error_attrs)
            vector_insert_duration.record(duration, metric_attrs)
            raise

    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        logger.debug("VectorIORouter.openai_list_files_in_vector_store", vector_store_id=vector_store_id)
        return await self.routing_table.openai_list_files_in_vector_store(
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
        logger.debug(
            "VectorIORouter.openai_retrieve_vector_store_file", vector_store_id=vector_store_id, file_id=file_id
        )
        return await self.routing_table.openai_retrieve_vector_store_file(
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
        logger.debug(
            "VectorIORouter.openai_retrieve_vector_store_file_contents: , , include_embeddings=, include_metadata",
            vector_store_id=vector_store_id,
            file_id=file_id,
            include_embeddings=include_embeddings,
            include_metadata=include_metadata,
        )

        return await self.routing_table.openai_retrieve_vector_store_file_contents(
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
        logger.debug("VectorIORouter.openai_update_vector_store_file", vector_store_id=vector_store_id, file_id=file_id)
        return await self.routing_table.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=request,
        )

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        logger.debug("VectorIORouter.openai_delete_vector_store_file", vector_store_id=vector_store_id, file_id=file_id)
        provider_id = self._get_provider_id(vector_store_id)
        metric_attrs = create_vector_metric_attributes(
            vector_db=vector_store_id,
            operation="file",
            provider=provider_id,
        )
        try:
            result = await self.routing_table.openai_delete_vector_store_file(
                vector_store_id=vector_store_id,
                file_id=file_id,
            )
            vector_deletes_total.add(1, {**metric_attrs, "status": "success"})
            return result
        except asyncio.CancelledError:
            vector_deletes_total.add(1, {**metric_attrs, "status": "error"})
            raise
        except Exception:
            vector_deletes_total.add(1, {**metric_attrs, "status": "error"})
            raise

    async def health(self) -> dict[str, HealthResponse]:
        health_statuses = {}
        timeout = 1  # increasing the timeout to 1 second for health checks
        for provider_id, impl in self.routing_table.impls_by_provider_id.items():
            try:
                # check if the provider has a health method
                if not hasattr(impl, "health"):
                    continue
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                health_statuses[provider_id] = health
            except TimeoutError:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR,
                    message=f"Health check timed out after {timeout} seconds",
                )
            except NotImplementedError:
                health_statuses[provider_id] = HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
            except Exception as e:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"
                )
        return health_statuses

    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        logger.debug(
            "VectorIORouter.openai_create_vector_store_file_batch: , files",
            vector_store_id=vector_store_id,
            file_ids_count=len(params.file_ids),
        )
        return await self.routing_table.openai_create_vector_store_file_batch(
            vector_store_id=vector_store_id,
            params=params,
        )

    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        logger.debug(
            "VectorIORouter.openai_retrieve_vector_store_file_batch", batch_id=batch_id, vector_store_id=vector_store_id
        )
        return await self.routing_table.openai_retrieve_vector_store_file_batch(
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
        logger.debug(
            "VectorIORouter.openai_list_files_in_vector_store_file_batch",
            batch_id=batch_id,
            vector_store_id=vector_store_id,
        )
        return await self.routing_table.openai_list_files_in_vector_store_file_batch(
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
        logger.debug(
            "VectorIORouter.openai_cancel_vector_store_file_batch", batch_id=batch_id, vector_store_id=vector_store_id
        )
        return await self.routing_table.openai_cancel_vector_store_file_batch(
            batch_id=batch_id,
            vector_store_id=vector_store_id,
        )
