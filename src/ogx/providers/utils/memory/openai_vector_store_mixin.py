# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import mimetypes
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Annotated, Any

from fastapi import Body

from ogx.core.datatypes import VectorStoresConfig
from ogx.core.id_generation import generate_object_id
from ogx.log import get_logger
from ogx.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from ogx.providers.utils.memory.vector_store import (
    content_from_data_and_mime_type,
    validate_tiktoken_encoding,
)
from ogx.providers.utils.vector_io.filters import parse_filter
from ogx_api import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    MAX_PAGINATION_LIMIT,
    Chunk,
    ChunkForDeletion,
    DeleteChunksRequest,
    EmbeddedChunk,
    FileProcessors,
    Files,
    Inference,
    InsertChunksRequest,
    OpenAIAttachFileRequest,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIFileObject,
    OpenAISearchVectorStoreRequest,
    OpenAISystemMessageParam,
    OpenAIUpdateVectorStoreFileRequest,
    OpenAIUpdateVectorStoreRequest,
    OpenAIUserMessageParam,
    QueryChunksRequest,
    QueryChunksResponse,
    SearchRankingOptions,
    TextContentItem,
    VectorStore,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyContextual,
    VectorStoreChunkingStrategyContextualConfig,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreContent,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileContentResponse,
    VectorStoreFileCounts,
    VectorStoreFileDeleteResponse,
    VectorStoreFileLastError,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreNotFoundError,
    VectorStoreObject,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)
from ogx_api.file_processors.models import ProcessFileRequest
from ogx_api.files.models import (
    RetrieveFileContentRequest,
    RetrieveFileRequest,
)
from ogx_api.internal.kvstore import KVStore

EMBEDDING_DIMENSION = 768

logger = get_logger(name=__name__, category="providers::utils")

# Constants for OpenAI vector stores

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:{VERSION}::"
OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX = f"openai_vector_stores_file_batches:{VERSION}::"


_RETRIABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


def _is_retriable_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "status_code", None)
    return status in _RETRIABLE_STATUS_CODES if status else False


class _ChunkContextResult(StrEnum):
    """Internal enum for chunk contextualization results."""

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


class OpenAIVectorStoreMixin(ABC):
    """
    Mixin class that provides common OpenAI Vector Store API implementation.
    Providers need to implement the abstract storage methods and maintain
    an openai_vector_stores in-memory cache.
    """

    # Implementing classes should call super().__init__() in their __init__ method
    # to properly initialize the mixin attributes.
    def __init__(
        self,
        inference_api: Inference,
        files_api: Files | None = None,
        kvstore: KVStore | None = None,
        vector_stores_config: VectorStoresConfig | None = None,
        file_processor_api: FileProcessors | None = None,
    ):
        if not inference_api:
            raise RuntimeError("Inference API is required for vector store operations")

        self.inference_api = inference_api
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}
        self.openai_file_batches: dict[str, dict[str, Any]] = {}
        self.files_api = files_api
        self.kvstore = kvstore
        self.vector_stores_config = vector_stores_config or VectorStoresConfig()
        self.file_processor_api = file_processor_api
        self._last_file_batch_cleanup_time = 0
        self._file_batch_tasks: dict[str, asyncio.Task[None]] = {}
        self._vector_store_locks: dict[str, asyncio.Lock] = {}

    def _get_vector_store_lock(self, vector_store_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific vector store."""
        if vector_store_id not in self._vector_store_locks:
            self._vector_store_locks[vector_store_id] = asyncio.Lock()
        return self._vector_store_locks[vector_store_id]

    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Save vector store metadata to persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))
        # update in-memory cache
        self.openai_vector_stores[store_id] = store_info

    async def _ensure_openai_metadata_exists(self, vector_store: VectorStore, name: str | None = None) -> None:
        """
        Ensure OpenAI-compatible metadata exists for a vector store.
        """
        if vector_store.identifier not in self.openai_vector_stores:
            store_info = {
                "id": vector_store.identifier,
                "object": "vector_store",
                "created_at": int(time.time()),
                "name": name or vector_store.vector_store_name or vector_store.identifier,
                "usage_bytes": 0,
                "file_counts": VectorStoreFileCounts(
                    cancelled=0,
                    completed=0,
                    failed=0,
                    in_progress=0,
                    total=0,
                ).model_dump(),
                "status": "completed",
                "expires_after": None,
                "expires_at": None,
                "last_active_at": int(time.time()),
                "file_ids": [],
                "chunking_strategy": None,
                "metadata": {
                    "provider_id": vector_store.provider_id,
                    "provider_vector_store_id": vector_store.provider_resource_id,
                    "embedding_model": vector_store.embedding_model,
                    "embedding_dimension": str(vector_store.embedding_dimension),
                },
            }
            await self._save_openai_vector_store(vector_store.identifier, store_info)

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load all vector store metadata from persistent storage."""
        assert self.kvstore
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored_data = await self.kvstore.values_in_range(start_key, end_key)

        stores: dict[str, dict[str, Any]] = {}
        for item in stored_data:
            info = json.loads(item)
            stores[info["id"]] = info
        return stores

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update vector store metadata in persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))
        # update in-memory cache
        self.openai_vector_stores[store_id] = store_info

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete vector store metadata from persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.delete(key)
        # remove from in-memory cache
        self.openai_vector_stores.pop(store_id, None)

    async def _save_openai_vector_store_file(
        self,
        store_id: str,
        file_id: str,
        file_info: dict[str, Any],
        file_contents: list[dict[str, Any]],
    ) -> None:
        """Save vector store file metadata to persistent storage."""
        assert self.kvstore
        meta_key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        await self.kvstore.set(key=meta_key, value=json.dumps(file_info))
        contents_prefix = f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}{store_id}:{file_id}:"
        for idx, chunk in enumerate(file_contents):
            await self.kvstore.set(key=f"{contents_prefix}{idx}", value=json.dumps(chunk))

    async def _load_openai_vector_store_file(self, store_id: str, file_id: str) -> dict[str, Any]:
        """Load vector store file metadata from persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        stored_data = await self.kvstore.get(key)
        return json.loads(stored_data) if stored_data else {}

    async def _load_openai_vector_store_file_contents(self, store_id: str, file_id: str) -> list[dict[str, Any]]:
        """Load vector store file contents from persistent storage."""
        assert self.kvstore
        prefix = f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}{store_id}:{file_id}:"
        end_key = f"{prefix}\xff"
        raw_items = await self.kvstore.values_in_range(prefix, end_key)
        return [json.loads(item) for item in raw_items]

    async def _update_openai_vector_store_file(self, store_id: str, file_id: str, file_info: dict[str, Any]) -> None:
        """Update vector store file metadata in persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        await self.kvstore.set(key=key, value=json.dumps(file_info))

    async def _delete_openai_vector_store_file_from_storage(self, store_id: str, file_id: str) -> None:
        """Delete vector store file metadata from persistent storage."""
        assert self.kvstore

        meta_key = f"{OPENAI_VECTOR_STORES_FILES_PREFIX}{store_id}:{file_id}"
        await self.kvstore.delete(meta_key)

        contents_prefix = f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}{store_id}:{file_id}:"
        end_key = f"{contents_prefix}\xff"
        # load all stored chunk values (values_in_range is implemented by all backends)
        raw_items = await self.kvstore.values_in_range(contents_prefix, end_key)
        # delete each chunk by its index suffix
        for idx in range(len(raw_items)):
            await self.kvstore.delete(f"{contents_prefix}{idx}")

    async def _save_openai_vector_store_file_batch(self, batch_id: str, batch_info: dict[str, Any]) -> None:
        """Save file batch metadata to persistent storage."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch_id}"
        await self.kvstore.set(key=key, value=json.dumps(batch_info))
        # update in-memory cache
        self.openai_file_batches[batch_id] = batch_info

    async def _load_openai_vector_store_file_batches(self) -> dict[str, dict[str, Any]]:
        """Load all file batch metadata from persistent storage."""
        assert self.kvstore
        start_key = OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}\xff"
        stored_data = await self.kvstore.values_in_range(start_key, end_key)

        batches: dict[str, dict[str, Any]] = {}
        for item in stored_data:
            info = json.loads(item)
            batches[info["id"]] = info
        return batches

    async def _delete_openai_vector_store_file_batch(self, batch_id: str) -> None:
        """Delete file batch metadata from persistent storage and in-memory cache."""
        assert self.kvstore
        key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch_id}"
        await self.kvstore.delete(key)
        # remove from in-memory cache
        self.openai_file_batches.pop(batch_id, None)

    async def _cleanup_expired_file_batches(self) -> None:
        """Clean up expired file batches from persistent storage."""
        assert self.kvstore
        start_key = OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}\xff"
        stored_data = await self.kvstore.values_in_range(start_key, end_key)

        current_time = int(time.time())
        expired_count = 0

        for item in stored_data:
            info = json.loads(item)
            expires_at = info.get("expires_at")
            if expires_at and current_time > expires_at:
                logger.info("Cleaning up expired file batch", id=info["id"])
                await self.kvstore.delete(f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{info['id']}")
                # Remove from in-memory cache if present
                self.openai_file_batches.pop(info["id"], None)
                expired_count += 1

        if expired_count > 0:
            logger.info("Cleaned up expired file batches", expired_count=expired_count)

    async def _get_processed_files_in_batch(
        self, vector_store_id: str, file_ids: list[str]
    ) -> tuple[set[str], set[str]]:
        """Determine which files in a batch are completed or failed.

        Returns:
            Tuple of (completed_file_ids, failed_file_ids).
        """
        if vector_store_id not in self.openai_vector_stores:
            return set(), set()

        store_info = self.openai_vector_stores[vector_store_id]
        known_file_ids = set(file_ids) & set(store_info["file_ids"])

        completed = set()
        failed = set()
        for file_id in known_file_ids:
            file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
            if file_info and file_info.get("status") == "failed":
                failed.add(file_id)
            else:
                completed.add(file_id)

        return completed, failed

    async def _analyze_batch_completion_on_resume(self, batch_id: str, batch_info: dict[str, Any]) -> list[str]:
        """Analyze batch completion status and return remaining files to process.

        Returns:
            List of file IDs that still need processing. Empty list if batch is complete.
        """
        vector_store_id = batch_info["vector_store_id"]
        all_file_ids = batch_info["file_ids"]

        # Find files that are completed or failed
        completed_files, failed_files = await self._get_processed_files_in_batch(vector_store_id, all_file_ids)
        processed_files = completed_files | failed_files
        remaining_files = [file_id for file_id in all_file_ids if file_id not in processed_files]

        completed_count = len(completed_files)
        failed_count = len(failed_files)
        total_count = len(all_file_ids)
        remaining_count = len(remaining_files)

        # Update file counts to reflect actual state
        batch_info["file_counts"] = {
            "completed": completed_count,
            "failed": failed_count,
            "in_progress": remaining_count,
            "cancelled": 0,
            "total": total_count,
        }

        # If all files are processed (completed or failed), mark batch as done
        if remaining_count == 0:
            batch_info["status"] = "completed"
            logger.info("Batch is already fully processed, updating status", batch_id=batch_id)

        # Save updated batch info
        await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        return remaining_files

    async def _resume_incomplete_batches(self) -> None:
        """Resume processing of incomplete file batches after server restart."""
        for batch_id, batch_info in self.openai_file_batches.items():
            if batch_info["status"] == "in_progress":
                logger.info("Analyzing incomplete file batch", batch_id=batch_id)

                remaining_files = await self._analyze_batch_completion_on_resume(batch_id, batch_info)

                # Check if batch is now completed after analysis
                if batch_info["status"] == "completed":
                    continue

                if remaining_files:
                    logger.info(
                        "Resuming batch with remaining files",
                        batch_id=batch_id,
                        remaining_files_count=len(remaining_files),
                    )
                    # Restart the background processing task with only remaining files
                    task = asyncio.create_task(self._process_file_batch_async(batch_id, batch_info, remaining_files))
                    self._file_batch_tasks[batch_id] = task

    async def initialize_openai_vector_stores(self) -> None:
        """Load existing OpenAI vector stores and file batches into the in-memory cache."""
        validate_tiktoken_encoding()
        if not self.files_api:
            logger.warning(
                "Files API is not available. File attachment operations on vector stores will fail. "
                "Ensure a 'files' provider is configured if file operations are needed."
            )
        self.openai_vector_stores = await self._load_openai_vector_stores()
        self.openai_file_batches = await self._load_openai_vector_store_file_batches()
        self._file_batch_tasks = {}
        # TODO: Resume only works for single worker deployment. Jobs with multiple workers will need to be handled differently.
        await self._resume_incomplete_batches()
        self._last_file_batch_cleanup_time = 0

    async def shutdown(self) -> None:
        """Clean up mixin resources including background tasks."""
        # Cancel any running file batch tasks gracefully
        tasks_to_cancel = list(self._file_batch_tasks.items())
        for _, task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @abstractmethod
    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Delete chunks from a vector store."""
        pass

    @abstractmethod
    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """Register a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Unregister a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def insert_chunks(
        self,
        request: InsertChunksRequest,
    ) -> None:
        """Insert chunks into a vector database (provider-specific implementation)."""
        pass

    @abstractmethod
    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        """Query chunks from a vector database (provider-specific implementation)."""
        pass

    async def openai_create_vector_store(
        self,
        params: Annotated[OpenAICreateVectorStoreRequestWithExtraBody, Body(...)],
    ) -> VectorStoreObject:
        """Creates a vector store."""
        created_at = int(time.time())

        # Extract ogx-specific parameters from extra_body
        extra_body = params.model_extra or {}
        metadata = params.metadata or {}

        provider_vector_store_id = extra_body.get("provider_vector_store_id")

        # Use embedding info from metadata if available, otherwise from extra_body
        if metadata.get("embedding_model"):
            # If either is in metadata, use metadata as source
            embedding_model = metadata.get("embedding_model")
            embedding_dimension = int(metadata["embedding_dimension"]) if metadata.get("embedding_dimension") else None
            logger.debug(
                "Using embedding config from metadata (takes precedence over extra_body): model=, dimension",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        else:
            embedding_model = extra_body.get("embedding_model")
            embedding_dimension = extra_body.get("embedding_dimension")
            logger.debug(
                "Using embedding config from extra_body: model=, dimension",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )

        # use provider_id set by router; fallback to provider's own ID when used directly via --stack-config
        provider_id = extra_body.get("provider_id") or getattr(self, "__provider_id__", None)
        # Derive the canonical vector_store_id (allow override, else generate)
        vector_store_id = provider_vector_store_id or generate_object_id("vector_store", lambda: f"vs_{uuid.uuid4()}")

        if embedding_model is None:
            raise ValueError("embedding_model is required")

        if embedding_dimension is None:
            raise ValueError(
                "Embedding dimension is required. Please provide 'embedding_dimension' in the request, "
                "or ensure the request goes through the router which can look it up from model metadata."
            )

        if provider_id is None:
            raise ValueError("Provider ID is required but was not provided")

        # Create OpenAI vector store metadata
        status = "completed"

        # Start with no files attached and update later
        file_counts = VectorStoreFileCounts(
            cancelled=0,
            completed=0,
            failed=0,
            in_progress=0,
            total=0,
        )
        if not params.chunking_strategy or params.chunking_strategy.type == "auto":
            chunking_strategy: VectorStoreChunkingStrategy = VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig(
                    max_chunk_size_tokens=DEFAULT_CHUNK_SIZE_TOKENS,
                    chunk_overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
                )
            )
        else:
            chunking_strategy = params.chunking_strategy
        store_info: dict[str, Any] = {
            "id": vector_store_id,
            "object": "vector_store",
            "created_at": created_at,
            "name": params.name or "",
            "usage_bytes": 0,
            "file_counts": file_counts.model_dump(),
            "status": status,
            "expires_after": params.expires_after.model_dump() if params.expires_after else None,
            "expires_at": None,
            "last_active_at": created_at,
            "file_ids": [],
            "chunking_strategy": chunking_strategy.model_dump(),
        }

        # Add provider information to metadata if provided
        if provider_id:
            metadata["provider_id"] = provider_id
        if provider_vector_store_id:
            metadata["provider_vector_store_id"] = provider_vector_store_id

        # Add embedding configuration to metadata for file processing
        metadata["embedding_model"] = embedding_model
        metadata["embedding_dimension"] = str(embedding_dimension)

        store_info["metadata"] = metadata

        # Save to persistent storage (provider-specific)
        await self._save_openai_vector_store(vector_store_id, store_info)

        # Store in memory cache
        self.openai_vector_stores[vector_store_id] = store_info

        # Now that our vector store is created, attach any files that were provided
        file_ids = params.file_ids or []
        tasks = [
            self.openai_attach_file_to_vector_store(vector_store_id, OpenAIAttachFileRequest(file_id=file_id))
            for file_id in file_ids
        ]
        # Use return_exceptions=True to handle individual file attachment failures gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions but don't fail the vector store creation
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to attach file to vector store",
                    file_ids_i=file_ids[i],
                    vector_store_id=vector_store_id,
                    result=result,
                )

        # Get the updated store info and return it
        store_info = self.openai_vector_stores[vector_store_id]
        return VectorStoreObject.model_validate(store_info)

    async def openai_list_vector_stores(
        self,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        """Returns a list of vector stores."""
        limit = min(limit or 20, MAX_PAGINATION_LIMIT)
        order = order or "desc"

        # Get all vector stores
        all_stores = list(self.openai_vector_stores.values())

        # Sort by created_at
        reverse_order = order == "desc"
        all_stores.sort(key=lambda x: x["created_at"], reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, store in enumerate(all_stores) if store["id"] == after), -1)
            if after_index >= 0:
                all_stores = all_stores[after_index + 1 :]

        if before:
            before_index = next(
                (i for i, store in enumerate(all_stores) if store["id"] == before),
                len(all_stores),
            )
            all_stores = all_stores[:before_index]

        # Apply limit
        limited_stores = all_stores[:limit]
        # Convert to VectorStoreObject instances
        data = [VectorStoreObject(**store) for store in limited_stores]

        # Determine pagination info
        has_more = len(all_stores) > limit
        first_id = data[0].id if data else ""
        last_id = data[-1].id if data else ""

        return VectorStoreListResponse(
            data=data,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]
        return VectorStoreObject(**store_info)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIUpdateVectorStoreRequest,
    ) -> VectorStoreObject:
        """Modifies a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id].copy()

        # Update fields if provided
        if request.name is not None:
            store_info["name"] = request.name
        if request.expires_after is not None:
            store_info["expires_after"] = (
                request.expires_after.model_dump()
                if hasattr(request.expires_after, "model_dump")
                else request.expires_after
            )
        if request.metadata is not None:
            store_info["metadata"] = request.metadata

        # Update last_active_at
        store_info["last_active_at"] = int(time.time())

        # Save to persistent storage (provider-specific)
        await self._update_openai_vector_store(vector_store_id, store_info)

        # Update in-memory cache
        self.openai_vector_stores[vector_store_id] = store_info

        return VectorStoreObject(**store_info)

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        # Delete from persistent storage (provider-specific)
        await self._delete_openai_vector_store_from_storage(vector_store_id)

        # Delete from in-memory cache
        self.openai_vector_stores.pop(vector_store_id, None)

        # Also delete the underlying vector DB
        try:
            await self.unregister_vector_store(vector_store_id)
        except Exception as e:
            logger.warning("Failed to delete underlying vector DB", vector_store_id=vector_store_id, error=str(e))

        return VectorStoreDeleteResponse(
            id=vector_store_id,
            deleted=True,
        )

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        request: OpenAISearchVectorStoreRequest,
    ) -> VectorStoreSearchResponsePage:
        """Search for chunks in a vector store.

        Note: Query rewriting is handled at the router level, not here.
        The rewrite_query parameter is kept for API compatibility but is ignored.
        """
        max_num_results = request.max_num_results or 10

        # Validate search_mode
        valid_modes = {"keyword", "vector", "hybrid"}
        if request.search_mode not in valid_modes:
            raise ValueError(f"search_mode must be one of {valid_modes}, got {request.search_mode}")

        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        if isinstance(request.query, list):
            search_query = " ".join(request.query)
        else:
            search_query = request.query

        try:
            # Validate neural ranker requires model parameter
            if request.ranking_options is not None:
                if getattr(request.ranking_options, "ranker", None) == "neural":
                    model_value = getattr(request.ranking_options, "model", None)
                    if model_value is None or (isinstance(model_value, str) and model_value.strip() == ""):
                        # Return empty results when model is missing for neural ranker
                        logger.warning("model parameter is required when ranker='neural', returning empty results")
                        return VectorStoreSearchResponsePage(
                            search_query=request.query if isinstance(request.query, list) else [request.query],
                            data=[],
                            has_more=False,
                            next_page=None,
                        )
            score_threshold = (
                request.ranking_options.score_threshold
                if request.ranking_options and request.ranking_options.score_threshold is not None
                else 0.0
            )
            params = {
                "max_num_results": max_num_results,
                "max_chunks": max_num_results * self.vector_stores_config.chunk_retrieval_params.chunk_multiplier,
                "score_threshold": score_threshold,
                "mode": request.search_mode,
            }

            # Parse filters into typed objects and pass through to the query
            if request.filters:
                params["filters"] = parse_filter(request.filters)

            # Use VectorStoresConfig defaults when ranking_options values are not provided
            config = self.vector_stores_config or VectorStoresConfig()
            params.update(self._build_reranker_params(request.ranking_options, config))

            response = await self.query_chunks(
                QueryChunksRequest(
                    vector_store_id=vector_store_id,
                    query=search_query,
                    params=params,
                )
            )

            # Convert response to OpenAI format
            data = []
            for embedded_chunk, score in zip(response.chunks, response.scores, strict=False):
                chunk = embedded_chunk
                content = self._chunk_to_vector_store_content(chunk)

                response_data_item = VectorStoreSearchResponse(
                    file_id=chunk.metadata.get("document_id", ""),
                    filename=chunk.metadata.get("filename", ""),
                    score=score,
                    attributes=chunk.metadata,
                    content=content,
                )
                data.append(response_data_item)
                if len(data) >= max_num_results:
                    break

            return VectorStoreSearchResponsePage(
                search_query=request.query if isinstance(request.query, list) else [request.query],
                data=data,
                has_more=False,  # For simplicity, we don't implement pagination here
                next_page=None,
            )

        except Exception as e:
            # Log the error and return empty results
            logger.error("Error searching vector store", vector_store_id=vector_store_id, error=str(e))
            return VectorStoreSearchResponsePage(
                search_query=request.query if isinstance(request.query, list) else [request.query],
                data=[],
                has_more=False,
                next_page=None,
            )

    def _build_reranker_params(
        self,
        ranking_options: SearchRankingOptions | None,
        config: VectorStoresConfig,
    ) -> dict[str, Any]:
        reranker_params: dict[str, Any] = {}
        params: dict[str, Any] = {}

        if ranking_options and ranking_options.ranker:
            reranker_type = ranking_options.ranker

            if ranking_options.ranker == "weighted":
                alpha = ranking_options.alpha
                if alpha is None:
                    alpha = config.chunk_retrieval_params.weighted_search_alpha
                reranker_params["alpha"] = alpha
                if ranking_options.weights:
                    reranker_params["weights"] = ranking_options.weights
            elif ranking_options.ranker == "rrf":
                # For RRF ranker, use impact_factor from request if provided, otherwise use VectorStoresConfig default
                impact_factor = ranking_options.impact_factor
                if impact_factor is None:
                    impact_factor = config.chunk_retrieval_params.rrf_impact_factor
                reranker_params["impact_factor"] = impact_factor
                # If weights dict is provided (for neural combination), store it
                if ranking_options.weights:
                    reranker_params["weights"] = ranking_options.weights
            elif ranking_options.ranker == "neural":
                reranker_params["model"] = ranking_options.model
            else:
                logger.debug("Unknown ranker value, passing through", ranker=ranking_options.ranker)

            params["reranker_type"] = reranker_type
            params["reranker_params"] = reranker_params

            # Store model and weights for neural reranking
            if ranking_options.model:
                params["neural_model"] = ranking_options.model
            if ranking_options.weights:
                params["neural_weights"] = ranking_options.weights
        elif ranking_options is None or ranking_options.ranker is None:
            # No ranker specified in request - use VectorStoresConfig default
            default_strategy = config.chunk_retrieval_params.default_reranker_strategy
            if default_strategy in ("weighted", "rrf"):
                params["reranker_type"] = default_strategy
                reranker_params = {}

                if default_strategy == "weighted":
                    reranker_params["alpha"] = config.chunk_retrieval_params.weighted_search_alpha
                elif default_strategy == "rrf":
                    reranker_params["impact_factor"] = config.chunk_retrieval_params.rrf_impact_factor

                params["reranker_params"] = reranker_params

        return params

    def _chunk_to_vector_store_content(
        self, chunk: EmbeddedChunk, include_embeddings: bool = False, include_metadata: bool = False
    ) -> list[VectorStoreContent]:
        def extract_fields() -> dict:
            """Extract metadata fields from chunk based on include flags."""
            return {
                "chunk_metadata": chunk.chunk_metadata if include_metadata else None,
                "metadata": chunk.metadata if include_metadata else None,
                "embedding": chunk.embedding if include_embeddings else None,
            }

        fields = extract_fields()

        if isinstance(chunk.content, str):
            content_item = VectorStoreContent(type="text", text=chunk.content, **fields)
            content = [content_item]
        elif isinstance(chunk.content, list):
            # TODO: Add support for other types of content
            content = []
            for item in chunk.content:
                if item.type == "text":
                    content_item = VectorStoreContent(type="text", text=item.text, **fields)
                    content.append(content_item)
        else:
            if chunk.content.type != "text":
                raise ValueError(f"Unsupported content type: {chunk.content.type}")

            content_item = VectorStoreContent(type="text", text=chunk.content.text, **fields)
            content = [content_item]
        return content

    async def openai_attach_file_to_vector_store(
        self,
        vector_store_id: str,
        request: OpenAIAttachFileRequest,
    ) -> VectorStoreFileObject:
        file_id = request.file_id
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        # Check if file is already attached to this vector store
        store_info = self.openai_vector_stores[vector_store_id]
        if file_id in store_info["file_ids"]:
            logger.warning(
                "File is already attached to vector store, skipping", file_id=file_id, vector_store_id=vector_store_id
            )
            # Return existing file object
            file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
            return VectorStoreFileObject(**file_info)

        attributes = request.attributes or {}
        chunking_strategy = request.chunking_strategy or VectorStoreChunkingStrategyAuto()
        created_at = int(time.time())
        chunks: list[Chunk] = []
        embedded_chunks: list[EmbeddedChunk] = []
        file_response: OpenAIFileObject | None = None

        vector_store_file_object = VectorStoreFileObject(
            id=file_id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
            created_at=created_at,
            status="in_progress",
            usage_bytes=0,
            vector_store_id=vector_store_id,
        )

        if not self.files_api:
            vector_store_file_object.status = "failed"
            vector_store_file_object.last_error = VectorStoreFileLastError(
                code="server_error",
                message="Files API is not available",
            )
            return vector_store_file_object

        if isinstance(chunking_strategy, VectorStoreChunkingStrategyContextual):
            ctx = chunking_strategy.contextual
            if not ctx.model_id and not self.vector_stores_config.contextual_retrieval_params.model:
                raise ValueError(
                    "Failed to initialize contextual chunking: model_id is required. "
                    "Provide it in chunking_strategy.contextual or configure a default "
                    "in contextual_retrieval_params.model on the server."
                )

        try:
            file_response = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))

            # Get embedding model info from vector store metadata
            store_info = self.openai_vector_stores[vector_store_id]
            embedding_model = store_info["metadata"].get("embedding_model")
            embedding_dimension = store_info["metadata"].get("embedding_dimension")

            chunk_attributes = attributes.copy()
            chunk_attributes["filename"] = file_response.filename
            chunk_attributes["file_id"] = file_id

            if not self.file_processor_api:
                raise RuntimeError(
                    "FileProcessor API is required for file processing but is not configured. "
                    "Please ensure a file_processors provider is registered in your stack configuration."
                )

            logger.debug("Using FileProcessor API to process file", file_id=file_id)
            pf_resp = await self.file_processor_api.process_file(
                ProcessFileRequest(file_id=file_id, chunking_strategy=chunking_strategy)
            )

            chunks = []
            for chunk in pf_resp.chunks:
                # Enhance chunk metadata with file info and attributes
                enhanced_metadata = chunk.metadata.copy() if chunk.metadata else {}
                enhanced_metadata.update(chunk_attributes)

                # Ensure document_id consistency
                if chunk.chunk_metadata:
                    chunk.chunk_metadata.document_id = file_id

                # Create enhanced chunk
                enhanced_chunk = Chunk(
                    content=chunk.content,
                    chunk_id=chunk.chunk_id,
                    metadata=enhanced_metadata,
                    chunk_metadata=chunk.chunk_metadata,
                )
                chunks.append(enhanced_chunk)

            logger.debug("FileProcessor generated chunks for file", chunk_count=len(chunks), file_id=file_id)

            if isinstance(chunking_strategy, VectorStoreChunkingStrategyContextual):
                mime_type, _ = mimetypes.guess_type(file_response.filename)
                content_response = await self.files_api.openai_retrieve_file_content(
                    RetrieveFileContentRequest(file_id=file_id)
                )
                full_content = content_from_data_and_mime_type(content_response.body, mime_type)
                await self._execute_contextual_chunk_transformation(chunks, full_content, chunking_strategy.contextual)
            if not chunks:
                vector_store_file_object.status = "failed"
                vector_store_file_object.last_error = VectorStoreFileLastError(
                    code="server_error",
                    message="No chunks were generated from the file",
                )
            else:
                # Validate embedding model and dimension are available
                if not embedding_model:
                    raise RuntimeError(f"Vector store {vector_store_id} is not properly configured for file processing")
                if not embedding_dimension:
                    raise RuntimeError(f"Vector store {vector_store_id} is not properly configured for file processing")

                # Generate embeddings for all chunks before insertion

                # Prepare embedding request for all chunks
                params = OpenAIEmbeddingsRequestWithExtraBody(
                    model=embedding_model,
                    input=[interleaved_content_as_str(c.content) for c in chunks],
                    dimensions=embedding_dimension,
                )
                resp = await self.inference_api.openai_embeddings(params)

                # Create EmbeddedChunk instances from chunks and their embeddings
                for chunk, data in zip(chunks, resp.data, strict=False):
                    # Ensure embedding is a list of floats
                    embedding = data.embedding
                    if isinstance(embedding, str):
                        # Handle case where embedding might be returned as a string (shouldn't normally happen)
                        raise ValueError(f"Received string embedding instead of list: {embedding}")
                    embedded_chunk = EmbeddedChunk(
                        content=chunk.content,
                        chunk_id=chunk.chunk_id,
                        metadata=chunk.metadata,
                        chunk_metadata=chunk.chunk_metadata,
                        embedding=embedding,
                        embedding_model=embedding_model,
                        embedding_dimension=len(embedding),
                    )
                    embedded_chunks.append(embedded_chunk)

                await self.insert_chunks(
                    InsertChunksRequest(
                        vector_store_id=vector_store_id,
                        chunks=embedded_chunks,
                    )
                )
                vector_store_file_object.status = "completed"
        except Exception as e:
            logger.exception("Error attaching file to vector store")
            vector_store_file_object.status = "failed"
            vector_store_file_object.last_error = VectorStoreFileLastError(
                code="server_error",
                message=str(e),
            )

        # Save vector store file to persistent storage AFTER insert_chunks
        # so that chunks include the embeddings that were generated
        file_info = vector_store_file_object.model_dump()
        file_info["filename"] = file_response.filename if file_response else ""

        dict_chunks = [c.model_dump() for c in embedded_chunks]
        await self._save_openai_vector_store_file(vector_store_id, file_id, file_info, dict_chunks)

        # Update file_ids and file_counts in vector store metadata
        # Use lock to prevent race condition when multiple files are attached concurrently
        async with self._get_vector_store_lock(vector_store_id):
            store_info = self.openai_vector_stores[vector_store_id].copy()
            # Deep copy file_counts to avoid mutating shared dict
            store_info["file_counts"] = store_info["file_counts"].copy()
            store_info["file_ids"] = store_info["file_ids"].copy()
            store_info["file_ids"].append(file_id)
            store_info["file_counts"]["total"] += 1
            store_info["file_counts"][vector_store_file_object.status] += 1

            # Save updated vector store to persistent storage
            await self._save_openai_vector_store(vector_store_id, store_info)

            self.openai_vector_stores[vector_store_id] = store_info

        return vector_store_file_object

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
        limit = min(limit or 20, MAX_PAGINATION_LIMIT)
        order = order or "desc"

        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]

        file_objects: list[VectorStoreFileObject] = []
        for file_id in store_info["file_ids"]:
            file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
            file_object = VectorStoreFileObject(**file_info)
            if filter and file_object.status != filter:
                continue
            file_objects.append(file_object)

        # Sort by created_at
        reverse_order = order == "desc"
        file_objects.sort(key=lambda x: x.created_at, reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, file in enumerate(file_objects) if file.id == after), -1)
            if after_index >= 0:
                file_objects = file_objects[after_index + 1 :]

        if before:
            before_index = next(
                (i for i, file in enumerate(file_objects) if file.id == before),
                len(file_objects),
            )
            file_objects = file_objects[:before_index]

        # Apply limit
        limited_files = file_objects[:limit]

        # Determine pagination info
        has_more = len(file_objects) > limit
        first_id = limited_files[0].id if file_objects else ""
        last_id = limited_files[-1].id if file_objects else ""

        return VectorStoreListFilesResponse(
            data=limited_files,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        """Retrieves a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]
        if file_id not in store_info["file_ids"]:
            raise ValueError(f"File {file_id} not found in vector store {vector_store_id}")

        file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
        return VectorStoreFileObject(**file_info)

    async def openai_retrieve_vector_store_file_contents(
        self,
        vector_store_id: str,
        file_id: str,
        include_embeddings: bool | None = False,
        include_metadata: bool | None = False,
    ) -> VectorStoreFileContentResponse:
        """Retrieves the contents of a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        # Parameters are already provided directly
        # include_embeddings and include_metadata are now function parameters

        dict_chunks = await self._load_openai_vector_store_file_contents(vector_store_id, file_id)
        chunks = [EmbeddedChunk.model_validate(c) for c in dict_chunks]
        content = []
        for chunk in chunks:
            content.extend(
                self._chunk_to_vector_store_content(
                    chunk, include_embeddings=include_embeddings or False, include_metadata=include_metadata or False
                )
            )
        return VectorStoreFileContentResponse(
            data=content,
            has_more=False,
        )

    async def openai_update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        request: OpenAIUpdateVectorStoreFileRequest,
    ) -> VectorStoreFileObject:
        """Updates a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        store_info = self.openai_vector_stores[vector_store_id]
        if file_id not in store_info["file_ids"]:
            raise ValueError(f"File {file_id} not found in vector store {vector_store_id}")

        file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
        file_info["attributes"] = request.attributes
        await self._update_openai_vector_store_file(vector_store_id, file_id, file_info)
        return VectorStoreFileObject(**file_info)

    async def openai_delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileDeleteResponse:
        """Deletes a vector store file."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        dict_chunks = await self._load_openai_vector_store_file_contents(vector_store_id, file_id)
        chunks = [Chunk.model_validate(c) for c in dict_chunks]

        # Create ChunkForDeletion objects with both chunk_id and document_id
        chunks_for_deletion = []
        for c in chunks:
            if c.chunk_id:
                document_id = c.metadata.get("document_id") or (
                    c.chunk_metadata.document_id if c.chunk_metadata else None
                )
                if document_id:
                    chunks_for_deletion.append(ChunkForDeletion(chunk_id=str(c.chunk_id), document_id=document_id))
                else:
                    logger.warning("Chunk has no document_id, skipping deletion", chunk_id=c.chunk_id)

        if chunks_for_deletion:
            await self.delete_chunks(
                DeleteChunksRequest(
                    vector_store_id=vector_store_id,
                    chunks=chunks_for_deletion,
                )
            )

        store_info = self.openai_vector_stores[vector_store_id].copy()

        file = await self.openai_retrieve_vector_store_file(vector_store_id, file_id)
        await self._delete_openai_vector_store_file_from_storage(vector_store_id, file_id)

        # Update in-memory cache
        store_info["file_ids"].remove(file_id)
        store_info["file_counts"][file.status] -= 1
        store_info["file_counts"]["total"] -= 1
        self.openai_vector_stores[vector_store_id] = store_info

        # Save updated vector store to persistent storage
        await self._save_openai_vector_store(vector_store_id, store_info)

        return VectorStoreFileDeleteResponse(
            id=file_id,
            deleted=True,
        )

    async def openai_create_vector_store_file_batch(
        self,
        vector_store_id: str,
        params: Annotated[OpenAICreateVectorStoreFileBatchRequestWithExtraBody, Body(...)],
    ) -> VectorStoreFileBatchObject:
        """Create a vector store file batch."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        chunking_strategy = params.chunking_strategy or VectorStoreChunkingStrategyAuto()

        created_at = int(time.time())
        batch_id = generate_object_id("vector_store_file_batch", lambda: f"batch_{uuid.uuid4()}")
        # File batches expire after 7 days
        expires_at = created_at + (7 * 24 * 60 * 60)

        # Initialize batch file counts - all files start as in_progress
        file_counts = VectorStoreFileCounts(
            completed=0,
            cancelled=0,
            failed=0,
            in_progress=len(params.file_ids),
            total=len(params.file_ids),
        )

        # Create batch object immediately with in_progress status
        batch_object = VectorStoreFileBatchObject(
            id=batch_id,
            created_at=created_at,
            vector_store_id=vector_store_id,
            status="in_progress",
            file_counts=file_counts,
        )

        batch_info = {
            **batch_object.model_dump(),
            "file_ids": params.file_ids,
            "attributes": params.attributes,
            "chunking_strategy": chunking_strategy.model_dump(),
            "expires_at": expires_at,
        }
        await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        # Start background processing of files
        task = asyncio.create_task(self._process_file_batch_async(batch_id, batch_info))
        self._file_batch_tasks[batch_id] = task

        # Run cleanup if needed (throttled to once every 1 day)
        current_time = int(time.time())
        if (
            current_time - self._last_file_batch_cleanup_time
            >= self.vector_stores_config.file_batch_params.cleanup_interval_seconds
        ):
            logger.info("Running throttled cleanup of expired file batches")
            asyncio.create_task(self._cleanup_expired_file_batches())
            self._last_file_batch_cleanup_time = current_time

        return batch_object

    async def _process_files_with_concurrency(
        self,
        file_ids: list[str],
        vector_store_id: str,
        attributes: dict[str, Any],
        chunking_strategy_obj: Any,
        batch_id: str,
        batch_info: dict[str, Any],
    ) -> None:
        """Process files with controlled concurrency and chunking."""
        semaphore = asyncio.Semaphore(self.vector_stores_config.file_batch_params.max_concurrent_files_per_batch)

        async def process_single_file(file_id: str) -> tuple[str, bool]:
            """Process a single file with concurrency control."""
            async with semaphore:
                try:
                    vector_store_file_object = await self.openai_attach_file_to_vector_store(
                        vector_store_id=vector_store_id,
                        request=OpenAIAttachFileRequest(
                            file_id=file_id,
                            attributes=attributes,
                            chunking_strategy=chunking_strategy_obj,
                        ),
                    )
                    return file_id, vector_store_file_object.status == "completed"
                except Exception as e:
                    logger.error("Failed to process file in batch", file_id=file_id, batch_id=batch_id, error=str(e))
                    return file_id, False

        # Process files in chunks to avoid creating too many tasks at once
        total_files = len(file_ids)
        chunk_size = self.vector_stores_config.file_batch_params.file_batch_chunk_size

        for i in range(0, total_files, chunk_size):
            chunk_file_ids = file_ids[i : i + chunk_size]
            tasks = [process_single_file(file_id) for file_id in chunk_file_ids]

            # Wait for this chunk of files to complete
            results = await asyncio.gather(*tasks)

            # Update batch info with results from this chunk
            completed_files = sum(1 for _, success in results if success)
            failed_files = sum(1 for _, success in results if not success)

            # Update batch info in storage
            batch_info["file_counts"]["completed"] += completed_files
            batch_info["file_counts"]["failed"] += failed_files
            batch_info["file_counts"]["in_progress"] -= len(results)

            await self._save_openai_vector_store_file_batch(batch_id, batch_info)

    async def _process_file_batch_async(
        self, batch_id: str, batch_info: dict[str, Any], file_ids_override: list[str] | None = None
    ) -> None:
        """Background task to process files in a batch."""
        try:
            vector_store_id = batch_info["vector_store_id"]
            file_ids = file_ids_override or batch_info["file_ids"]
            attributes = batch_info.get("attributes") or {}
            chunking_strategy_dict = batch_info.get("chunking_strategy")

            # Reconstruct chunking strategy object
            chunking_strategy_obj: VectorStoreChunkingStrategy | None = None
            if chunking_strategy_dict:
                strategy_type = chunking_strategy_dict.get("type")
                if strategy_type == "static":
                    chunking_strategy_obj = VectorStoreChunkingStrategyStatic(
                        static=VectorStoreChunkingStrategyStaticConfig(**chunking_strategy_dict["static"])
                    )
                elif strategy_type == "contextual":
                    chunking_strategy_obj = VectorStoreChunkingStrategyContextual(
                        contextual=VectorStoreChunkingStrategyContextualConfig(**chunking_strategy_dict["contextual"])
                    )
                else:
                    if strategy_type != "auto":
                        logger.warning(
                            "Unknown chunking strategy type, falling back to auto", strategy_type=strategy_type
                        )
                    chunking_strategy_obj = VectorStoreChunkingStrategyAuto()

            await self._process_files_with_concurrency(
                file_ids,
                vector_store_id,
                attributes,
                chunking_strategy_obj,
                batch_id,
                batch_info,
            )

            # Mark batch as completed
            batch_info["status"] = "completed"
            batch_info["file_counts"]["in_progress"] = 0
            await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        except asyncio.CancelledError:
            logger.info("File batch processing cancelled for batch", batch_id=batch_id)
            batch_info["status"] = "cancelled"
            await self._save_openai_vector_store_file_batch(batch_id, batch_info)
            raise
        except Exception:
            logger.exception("Failed to process file batch", batch_id=batch_id)
            batch_info["status"] = "failed"
            await self._save_openai_vector_store_file_batch(batch_id, batch_info)

    async def openai_retrieve_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Retrieve a vector store file batch."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        if batch_id not in self.openai_file_batches:
            raise ValueError(f"File batch {batch_id} not found")

        batch_info = self.openai_file_batches[batch_id]
        if batch_info["vector_store_id"] != vector_store_id:
            raise ValueError(f"File batch {batch_id} does not belong to vector store {vector_store_id}")

        # Check if batch has expired (7 days from creation)
        import time

        current_time = int(time.time())
        created_at = batch_info.get("created_at", current_time)
        expires_at = batch_info.get("expires_at", created_at + (7 * 24 * 60 * 60))  # 7 days default

        if current_time > expires_at:
            raise ValueError(f"File batch {batch_id} has expired after 7 days from creation")

        return VectorStoreFileBatchObject(**batch_info)

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
        limit = min(limit or 20, MAX_PAGINATION_LIMIT)
        order = order or "desc"

        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        if batch_id not in self.openai_file_batches:
            raise ValueError(f"File batch {batch_id} not found")

        batch_info = self.openai_file_batches[batch_id]
        if batch_info["vector_store_id"] != vector_store_id:
            raise ValueError(f"File batch {batch_id} does not belong to vector store {vector_store_id}")

        file_ids = batch_info["file_ids"]
        file_objects = []

        # This could be slow for large batches if we load every file object.
        # But we need to load them to filter/sort.
        # Ideally we would store files per batch in a way that is efficiently queryable.
        # For now, load them.
        for file_id in file_ids:
            try:
                file_info = await self._load_openai_vector_store_file(vector_store_id, file_id)
                if not file_info:
                    continue
                file_object = VectorStoreFileObject(**file_info)
                if filter and file_object.status != filter:
                    continue
                file_objects.append(file_object)
            except Exception:
                # File might have been deleted or failed to load
                continue

        # Sort by created_at
        reverse_order = order == "desc"
        file_objects.sort(key=lambda x: x.created_at, reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, file in enumerate(file_objects) if file.id == after), -1)
            if after_index >= 0:
                file_objects = file_objects[after_index + 1 :]

        if before:
            before_index = next(
                (i for i, file in enumerate(file_objects) if file.id == before),
                len(file_objects),
            )
            file_objects = file_objects[:before_index]

        # Apply limit
        limited_files = file_objects[:limit]

        # Determine pagination info
        has_more = len(file_objects) > limit
        first_id = limited_files[0].id if limited_files else ""
        last_id = limited_files[-1].id if limited_files else ""

        return VectorStoreFilesListInBatchResponse(
            data=limited_files,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_cancel_vector_store_file_batch(
        self,
        batch_id: str,
        vector_store_id: str,
    ) -> VectorStoreFileBatchObject:
        """Cancels a vector store file batch."""
        if vector_store_id not in self.openai_vector_stores:
            raise VectorStoreNotFoundError(vector_store_id)

        if batch_id not in self.openai_file_batches:
            raise ValueError(f"File batch {batch_id} not found")

        batch_info = self.openai_file_batches[batch_id]
        if batch_info["vector_store_id"] != vector_store_id:
            raise ValueError(f"File batch {batch_id} does not belong to vector store {vector_store_id}")

        if batch_info["status"] in ["completed", "failed"]:
            raise ValueError(f"Cannot cancel batch {batch_id} with status {batch_info['status']}")

        # Cancel the background task if running
        if batch_id in self._file_batch_tasks:
            task = self._file_batch_tasks[batch_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self._file_batch_tasks[batch_id]

        batch_info["status"] = "cancelled"
        await self._save_openai_vector_store_file_batch(batch_id, batch_info)

        return VectorStoreFileBatchObject(**batch_info)

    async def _execute_contextual_chunk_transformation(
        self,
        chunks: list[Chunk],
        full_content: str,
        strategy_config: VectorStoreChunkingStrategyContextualConfig,
    ) -> None:
        """
        Applies contextual chunk transformation (to support Contextual Retrieval) by situating
        each chunk within the overall document.

        NOTE: This method mutates the Chunk objects in-place by prepending context to their content.
        """
        ctx_config = self.vector_stores_config.contextual_retrieval_params

        model_id = strategy_config.model_id
        if not model_id:
            if ctx_config.model:
                model_id = f"{ctx_config.model.provider_id}/{ctx_config.model.model_id}"
            else:
                raise ValueError(
                    "Failed to initialize contextual chunking: model_id is required. "
                    "Provide it in chunking_strategy.contextual or configure a default "
                    "in contextual_retrieval_params.model on the server."
                )

        timeout_seconds = strategy_config.timeout_seconds or ctx_config.default_timeout_seconds
        max_concurrency = strategy_config.max_concurrency or ctx_config.default_max_concurrency

        doc_token_estimate = len(full_content) // 4
        if doc_token_estimate > ctx_config.max_document_tokens:
            raise ValueError(
                f"Failed to process document for contextual retrieval: size (~{doc_token_estimate} tokens) "
                f"exceeds maximum allowed ({ctx_config.max_document_tokens} tokens)"
            )

        context_prompt_template = strategy_config.context_prompt

        logger.info("Applying contextual retrieval to chunks using model", chunks_count=len(chunks), model_id=model_id)

        # Split prompt into system (document) + user (chunk) messages to enable prefix caching.
        # All major providers (OpenAI, vLLM, Anthropic) cache shared token prefixes by placing the
        # document in a system message, the KV-cache is computed once and reused across all chunk requests.
        chunk_placeholder = "{{CHUNK_CONTENT}}"
        split_idx = context_prompt_template.index(chunk_placeholder)
        system_template = context_prompt_template[:split_idx].replace("{{WHOLE_DOCUMENT}}", full_content).rstrip()
        user_template = context_prompt_template[split_idx:]

        semaphore = asyncio.Semaphore(max_concurrency)

        async def contextualize_chunk(chunk: Chunk) -> _ChunkContextResult:
            async with semaphore:
                user_prompt = user_template.replace(chunk_placeholder, interleaved_content_as_str(chunk.content))
                messages: list = [
                    OpenAISystemMessageParam(role="system", content=system_template),
                    OpenAIUserMessageParam(role="user", content=user_prompt),
                ]

                params = OpenAIChatCompletionRequestWithExtraBody(
                    model=model_id,
                    messages=messages,
                    stream=False,
                    temperature=0.0,
                    max_tokens=256,
                )
                for attempt in range(_MAX_RETRIES + 1):
                    try:
                        response = await asyncio.wait_for(
                            self.inference_api.openai_chat_completion(params), timeout=timeout_seconds
                        )

                        if isinstance(response, AsyncIterator):
                            raise TypeError(
                                f"Failed to contextualize chunk {chunk.chunk_id}: "
                                "received streaming response, contextual retrieval requires non-streaming inference."
                            )

                        if not response.choices:
                            raise ValueError(
                                f"Failed to contextualize chunk {chunk.chunk_id}: LLM returned empty choices"
                            )

                        raw_context = response.choices[0].message.content
                        if raw_context is None:
                            logger.warning("LLM returned None content for chunk", chunk_id=chunk.chunk_id)
                            return _ChunkContextResult.EMPTY
                        context = (
                            raw_context.strip()
                            if isinstance(raw_context, str)
                            else " ".join(
                                p.text for p in raw_context if isinstance(p, OpenAIChatCompletionContentPartTextParam)
                            ).strip()
                        )
                        if not context:
                            logger.warning("LLM returned empty context for chunk", chunk_id=chunk.chunk_id)
                            return _ChunkContextResult.EMPTY
                        if isinstance(chunk.content, str):
                            chunk.content = f"{context}\n\n{chunk.content}"
                        elif isinstance(chunk.content, list):
                            chunk.content.insert(0, TextContentItem(text=f"{context}\n\n"))
                        return _ChunkContextResult.SUCCESS

                    except asyncio.CancelledError:
                        raise
                    except (MemoryError, SystemExit, KeyboardInterrupt, RecursionError):
                        raise
                    except Exception as e:
                        if _is_retriable_error(e) and attempt < _MAX_RETRIES:
                            delay = _RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
                            logger.warning(
                                "Chunk : retriable error (), retrying in s (attempt /)",
                                chunk_id=chunk.chunk_id,
                                error_type=type(e).__name__,
                                delay=delay,
                                attempt_1=attempt + 1,
                                max_retries_1=_MAX_RETRIES + 1,
                            )
                            await asyncio.sleep(delay)
                            continue
                        if _is_retriable_error(e):
                            logger.error(
                                "Chunk : exhausted attempts, last error",
                                chunk_id=chunk.chunk_id,
                                max_retries_1=_MAX_RETRIES + 1,
                                error_type=type(e).__name__,
                                error=str(e),
                            )
                        else:
                            logger.error(
                                "Failed to contextualize chunk",
                                chunk_id=chunk.chunk_id,
                                error_type=type(e).__name__,
                                error=str(e),
                            )
                        return _ChunkContextResult.FAILED

                return _ChunkContextResult.FAILED

        results = await asyncio.gather(*(contextualize_chunk(chunk) for chunk in chunks))

        total = len(chunks)
        fail_count = sum(1 for r in results if r == _ChunkContextResult.FAILED)
        empty_count = sum(1 for r in results if r == _ChunkContextResult.EMPTY)

        if fail_count > 0:
            if fail_count == total:
                raise RuntimeError(f"Failed to contextualize any chunks for model {model_id}")
            logger.warning("Contextual retrieval partially failed: / chunks failed", fail_count=fail_count, total=total)

        if empty_count > 0:
            if empty_count == total and fail_count == 0:
                raise RuntimeError(
                    f"Failed to generate context using model {model_id}: empty context for all {total} chunks. "
                    "Verify the model supports instruction-following and the prompt template is appropriate."
                )
            logger.warning(
                "Contextual retrieval: / chunks received empty context", empty_count=empty_count, total=total
            )

        if fail_count == 0 and empty_count == 0:
            logger.info("Successfully contextualized all chunks", total=total)
