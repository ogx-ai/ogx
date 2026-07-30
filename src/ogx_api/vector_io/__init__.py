# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""VectorIO API module.

This module provides the VectorIO API for vector database operations.
Protocol definitions are in api.py, Pydantic models are in models.py,
and FastAPI routes are in fastapi_routes.py.
"""

# Re-export Protocol classes from api.py
from . import fastapi_routes
from .api import VectorIO, VectorStoreTable

# Re-export all Pydantic models from models.py
from .models import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    MAX_PAGINATION_LIMIT,
    Chunk,
    ChunkForDeletion,
    ChunkMetadata,
    DeleteChunksRequest,
    EmbeddedChunk,
    InsertChunksRequest,
    OpenAIAttachFileRequest,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAICreateVectorStoreRequestWithExtraBody,
    OpenAISearchVectorStoreRequest,
    OpenAIUpdateVectorStoreFileRequest,
    OpenAIUpdateVectorStoreRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorStoreChunkingStrategy,
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyContextual,
    VectorStoreChunkingStrategyContextualConfig,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreContent,
    VectorStoreCreateRequest,
    VectorStoreDeleteResponse,
    VectorStoreFileAttributes,
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
    VectorStoreModifyRequest,
    VectorStoreObject,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)

__all__ = [
    # Protocol classes
    "VectorIO",
    "VectorStoreTable",
    # Constants
    "DEFAULT_CHUNK_OVERLAP_TOKENS",
    "DEFAULT_CHUNK_SIZE_TOKENS",
    "MAX_PAGINATION_LIMIT",
    # Pydantic models
    "Chunk",
    "ChunkForDeletion",
    "ChunkMetadata",
    "DeleteChunksRequest",
    "EmbeddedChunk",
    "InsertChunksRequest",
    "OpenAIAttachFileRequest",
    "OpenAICreateVectorStoreFileBatchRequestWithExtraBody",
    "OpenAICreateVectorStoreRequestWithExtraBody",
    "OpenAISearchVectorStoreRequest",
    "OpenAIUpdateVectorStoreFileRequest",
    "OpenAIUpdateVectorStoreRequest",
    "QueryChunksRequest",
    "QueryChunksResponse",
    "SearchRankingOptions",
    "VectorStoreChunkingStrategy",
    "VectorStoreChunkingStrategyAuto",
    "VectorStoreChunkingStrategyContextual",
    "VectorStoreChunkingStrategyContextualConfig",
    "VectorStoreChunkingStrategyStatic",
    "VectorStoreChunkingStrategyStaticConfig",
    "VectorStoreContent",
    "VectorStoreCreateRequest",
    "VectorStoreDeleteResponse",
    "VectorStoreFileAttributes",
    "VectorStoreFileBatchObject",
    "VectorStoreFileContentResponse",
    "VectorStoreFileCounts",
    "VectorStoreFileDeleteResponse",
    "VectorStoreFileLastError",
    "VectorStoreFileObject",
    "VectorStoreFileStatus",
    "VectorStoreFilesListInBatchResponse",
    "VectorStoreListFilesResponse",
    "VectorStoreListResponse",
    "VectorStoreModifyRequest",
    "VectorStoreObject",
    "VectorStoreSearchRequest",
    "VectorStoreSearchResponse",
    "VectorStoreSearchResponsePage",
    "fastapi_routes",
]
