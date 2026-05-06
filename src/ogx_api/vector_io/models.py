# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for VectorIO API requests and responses.

This module defines the request and response models for the VectorIO API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ogx_api.common.content_types import InterleavedContent
from ogx_api.schema_utils import json_schema_type, register_schema

# OpenAI-compatible chunking defaults
# See: https://platform.openai.com/docs/api-reference/vector-stores-files/createFile
DEFAULT_CHUNK_SIZE_TOKENS = 800
DEFAULT_CHUNK_OVERLAP_TOKENS = 400

# Pagination limits matching OpenAI API constraints
MAX_PAGINATION_LIMIT = 100


@json_schema_type
class ChunkMetadata(BaseModel):
    """
    `ChunkMetadata` is backend metadata for a `Chunk` that is used to store additional information about the chunk that
        will not be used in the context during inference, but is required for backend functionality. The `ChunkMetadata`
        is set during chunk creation in `FileSearchToolRuntimeImpl().insert()`and is not expected to change after.
        Use `Chunk.metadata` for metadata that will be used in the context during inference.
    :param chunk_id: The ID of the chunk. If not set, it will be generated based on the document ID and content.
    :param document_id: The ID of the document this chunk belongs to.
    :param source: The source of the content, such as a URL, file path, or other identifier.
    :param created_timestamp: An optional timestamp indicating when the chunk was created.
    :param updated_timestamp: An optional timestamp indicating when the chunk was last updated.
    :param chunk_window: The window of the chunk, which can be used to group related chunks together.
    :param chunk_tokenizer: The tokenizer used to create the chunk. Default is Tiktoken.
    :param content_token_count: The number of tokens in the content of the chunk.
    :param metadata_token_count: The number of tokens in the metadata of the chunk.
    """

    chunk_id: str | None = None
    document_id: str | None = None
    source: str | None = None
    created_timestamp: int | None = None
    updated_timestamp: int | None = None
    chunk_window: str | None = None
    chunk_tokenizer: str | None = None
    content_token_count: int | None = None
    metadata_token_count: int | None = None


@json_schema_type
class Chunk(BaseModel):
    """
    A chunk of content from file processing.
    :param content: The content of the chunk, which can be interleaved text, images, or other types.
    :param chunk_id: Unique identifier for the chunk. Must be provided explicitly.
    :param metadata: Metadata associated with the chunk that will be used in the model context during inference.
    :param chunk_metadata: Metadata for the chunk that will NOT be used in the context during inference.
        The `chunk_metadata` is required backend functionality.
    """

    content: InterleavedContent
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_metadata: ChunkMetadata

    @property
    def document_id(self) -> str | None:
        """Returns the document_id from either metadata or chunk_metadata, with metadata taking precedence."""
        # Check metadata first (takes precedence)
        doc_id = self.metadata.get("document_id")
        if doc_id is not None:
            if not isinstance(doc_id, str):
                raise TypeError(f"metadata['document_id'] must be a string, got {type(doc_id).__name__}: {doc_id!r}")
            return doc_id

        # Fall back to chunk_metadata if available (Pydantic ensures type safety)
        if self.chunk_metadata is not None:
            return self.chunk_metadata.document_id

        return None


@json_schema_type
class EmbeddedChunk(Chunk):
    """
    A chunk of content with its embedding vector for vector database operations.
    Inherits all fields from Chunk and adds embedding-related fields.
    :param embedding: The embedding vector for the chunk content.
    :param embedding_model: The model used to generate the embedding (e.g., 'openai/text-embedding-3-small').
    :param embedding_dimension: The dimension of the embedding vector.
    """

    embedding: list[float]
    embedding_model: str
    embedding_dimension: int


@json_schema_type
class QueryChunksResponse(BaseModel):
    """Response from querying chunks in a vector database.

    :param chunks: List of embedded chunks returned from the query
    :param scores: Relevance scores corresponding to each returned chunk
    """

    chunks: list[EmbeddedChunk]
    scores: list[float]


@json_schema_type
class VectorStoreExpirationAfter(BaseModel):
    """Expiration policy for a vector store.

    :param anchor: Anchor timestamp after which the expiration policy applies. Supported anchors: last_active_at
    :param days: The number of days after the anchor time that the vector store will expire
    """

    anchor: Literal["last_active_at"] = Field(description="Anchor timestamp after which the expiration policy applies.")
    days: int = Field(
        ge=1, le=365, description="The number of days after the anchor time that the vector store will expire."
    )


@json_schema_type
class VectorStoreFileCounts(BaseModel):
    """File processing status counts for a vector store.

    :param completed: Number of files that have been successfully processed
    :param cancelled: Number of files that had their processing cancelled
    :param failed: Number of files that failed to process
    :param in_progress: Number of files currently being processed
    :param total: Total number of files in the vector store
    """

    completed: int
    cancelled: int
    failed: int
    in_progress: int
    total: int


VectorStoreStatus = Literal["expired", "in_progress", "completed"]
register_schema(VectorStoreStatus, name="VectorStoreStatus")


@json_schema_type
class VectorStoreObject(BaseModel):
    """OpenAI Vector Store object.

    :param id: Unique identifier for the vector store
    :param object: Object type identifier, always "vector_store"
    :param created_at: Timestamp when the vector store was created
    :param name: Name of the vector store
    :param usage_bytes: Storage space used by the vector store in bytes
    :param file_counts: File processing status counts for the vector store
    :param status: Current status of the vector store
    :param expires_after: (Optional) Expiration policy for the vector store
    :param expires_at: (Optional) Timestamp when the vector store will expire
    :param last_active_at: (Optional) Timestamp of last activity on the vector store
    :param metadata: Set of key-value pairs that can be attached to the vector store
    """

    id: str
    object: Literal["vector_store"] = "vector_store"
    created_at: int
    name: str = ""
    usage_bytes: int = 0
    file_counts: VectorStoreFileCounts
    status: VectorStoreStatus
    expires_after: VectorStoreExpirationAfter | None = None
    expires_at: int | None = None
    last_active_at: int | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreCreateRequest(BaseModel):
    """Request to create a vector store.

    :param name: (Optional) Name for the vector store
    :param description: (Optional) Description of the vector store
    :param file_ids: List of file IDs to include in the vector store
    :param expires_after: (Optional) Expiration policy for the vector store
    :param chunking_strategy: (Optional) Strategy for splitting files into chunks
    :param metadata: Set of key-value pairs that can be attached to the vector store
    """

    name: str | None = None
    description: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    expires_after: VectorStoreExpirationAfter | None = None
    chunking_strategy: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreModifyRequest(BaseModel):
    """Request to modify a vector store.

    :param name: (Optional) Updated name for the vector store
    :param expires_after: (Optional) Updated expiration policy for the vector store
    :param metadata: (Optional) Updated set of key-value pairs for the vector store
    """

    name: str | None = None
    expires_after: VectorStoreExpirationAfter | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreListResponse(BaseModel):
    """Response from listing vector stores.

    :param object: Object type identifier, always "list"
    :param data: List of vector store objects
    :param first_id: ID of the first vector store in the list for pagination
    :param last_id: ID of the last vector store in the list for pagination
    :param has_more: Whether there are more vector stores available beyond this page
    """

    object: Literal["list"] = "list"
    data: list[VectorStoreObject]
    first_id: str
    last_id: str
    has_more: bool


@json_schema_type
class VectorStoreSearchRequest(BaseModel):
    """Request to search a vector store.

    :param query: Search query as a string or list of strings
    :param filters: (Optional) Filters based on file attributes to narrow search results
    :param max_num_results: Maximum number of results to return, defaults to 10
    :param ranking_options: (Optional) Options for ranking and filtering search results
    :param rewrite_query: Whether to rewrite the query for better vector search performance
    """

    query: str | list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int = Field(default=10, ge=1, le=50)
    ranking_options: dict[str, Any] | None = None
    rewrite_query: bool = False


@json_schema_type
class VectorStoreContent(BaseModel):
    """Content item from a vector store file or search result.

    :param type: Content type, currently only "text" is supported
    :param text: The actual text content
    :param embedding: Optional embedding vector for this content chunk
    :param chunk_metadata: Optional chunk metadata
    :param metadata: Optional user-defined metadata
    """

    type: Literal["text"]
    text: str
    embedding: list[float] | None = None
    chunk_metadata: ChunkMetadata | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreSearchResponse(BaseModel):
    """Response from searching a vector store.

    :param file_id: Unique identifier of the file containing the result
    :param filename: Name of the file containing the result
    :param score: Relevance score for this search result
    :param attributes: (Optional) Key-value attributes associated with the file
    :param content: List of content items matching the search query
    """

    file_id: str
    filename: str
    score: float
    attributes: dict[str, str | float | bool] | None = None
    content: list[VectorStoreContent]


@json_schema_type
class VectorStoreSearchResponsePage(BaseModel):
    """Paginated response from searching a vector store.

    :param object: Object type identifier for the search results page
    :param search_query: The original search query that was executed
    :param data: List of search result objects
    :param has_more: Whether there are more results available beyond this page
    :param next_page: (Optional) Token for retrieving the next page of results
    """

    object: Literal["vector_store.search_results.page"] = "vector_store.search_results.page"
    search_query: list[str]
    data: list[VectorStoreSearchResponse]
    has_more: bool
    next_page: str | None = None


@json_schema_type
class VectorStoreDeleteResponse(BaseModel):
    """Response from deleting a vector store.

    :param id: Unique identifier of the deleted vector store
    :param object: Object type identifier for the deletion response
    :param deleted: Whether the deletion operation was successful
    """

    id: str
    object: Literal["vector_store.deleted"] = "vector_store.deleted"
    deleted: bool


@json_schema_type
class VectorStoreFileContentResponse(BaseModel):
    """Represents the parsed content of a vector store file.

    :param object: The object type, which is always `vector_store.file_content.page`
    :param data: Parsed content of the file
    :param has_more: Indicates if there are more content pages to fetch
    :param next_page: The token for the next page, if any
    """

    object: Literal["vector_store.file_content.page"] = "vector_store.file_content.page"
    data: list[VectorStoreContent]
    has_more: bool
    next_page: str | None = None


@json_schema_type
class VectorStoreChunkingStrategyAuto(BaseModel):
    """Automatic chunking strategy for vector store files.

    :param type: Strategy type, always "auto" for automatic chunking
    """

    type: Literal["auto"] = "auto"


@json_schema_type
class VectorStoreChunkingStrategyStaticConfig(BaseModel):
    """Configuration for static chunking strategy.

    :param chunk_overlap_tokens: Number of tokens to overlap between adjacent chunks
    :param max_chunk_size_tokens: Maximum number of tokens per chunk, must be between 100 and 4096
    """

    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    max_chunk_size_tokens: int = Field(DEFAULT_CHUNK_SIZE_TOKENS, ge=100, le=4096)


@json_schema_type
class VectorStoreChunkingStrategyStatic(BaseModel):
    """Static chunking strategy with configurable parameters.

    :param type: Strategy type, always "static" for static chunking
    :param static: Configuration parameters for the static chunking strategy
    """

    type: Literal["static"] = "static"
    static: VectorStoreChunkingStrategyStaticConfig


DEFAULT_CONTEXT_PROMPT = (
    "<document>\n{{WHOLE_DOCUMENT}}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{{CHUNK_CONTENT}}\n</chunk>\n"
    "Please give a short succinct description to situate this chunk of text within the overall document "
    "for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct description and nothing else."
)


def _strip_context_prompt_default(schema: dict) -> None:
    """Strip context_prompt default from JSON schema to prevent double-curly-brace
    template placeholders from breaking Stainless SDK code generation."""
    if props := schema.get("properties", {}):
        if cp := props.get("context_prompt"):
            cp.pop("default", None)


@json_schema_type
class VectorStoreChunkingStrategyContextualConfig(BaseModel):
    """Configuration for contextual chunking that uses an LLM to situate chunks within the document."""

    model_config = ConfigDict(json_schema_extra=_strip_context_prompt_default)

    model_id: str | None = Field(
        default=None,
        min_length=1,
        description="LLM model for generating context. Falls back to VectorStoresConfig.contextual_retrieval_params.model if not provided.",
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Prompt template for contextual retrieval. Uses WHOLE_DOCUMENT and CHUNK_CONTENT placeholders wrapped in double curly braces.",
    )
    max_chunk_size_tokens: int = Field(
        default=700,
        ge=100,
        le=4096,
        description="Maximum tokens per chunk. Suggested ~700 to allow room for prepended context.",
    )
    chunk_overlap_tokens: int = Field(
        default=400,
        ge=0,
        description="Tokens to overlap between adjacent chunks. Must be less than max_chunk_size_tokens.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Timeout per LLM call in seconds. Falls back to config default if not provided.",
    )
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description="Maximum concurrent LLM calls. Falls back to config default if not provided.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "VectorStoreChunkingStrategyContextualConfig":
        if self.chunk_overlap_tokens >= self.max_chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be less than max_chunk_size_tokens")

        if "{{WHOLE_DOCUMENT}}" not in self.context_prompt:
            raise ValueError("context_prompt must contain {{WHOLE_DOCUMENT}} placeholder")
        if "{{CHUNK_CONTENT}}" not in self.context_prompt:
            raise ValueError("context_prompt must contain {{CHUNK_CONTENT}} placeholder")
        if self.context_prompt.index("{{WHOLE_DOCUMENT}}") >= self.context_prompt.index("{{CHUNK_CONTENT}}"):
            raise ValueError(
                "context_prompt must have {{WHOLE_DOCUMENT}} before {{CHUNK_CONTENT}} to enable prefix caching"
            )

        return self


@json_schema_type
class VectorStoreChunkingStrategyContextual(BaseModel):
    """Contextual chunking strategy that uses an LLM to situate chunks within the document."""

    type: Literal["contextual"] = Field(default="contextual", description="Strategy type identifier.")
    contextual: VectorStoreChunkingStrategyContextualConfig = Field(
        description="Configuration for contextual chunking."
    )


VectorStoreChunkingStrategy = Annotated[
    VectorStoreChunkingStrategyAuto | VectorStoreChunkingStrategyStatic | VectorStoreChunkingStrategyContextual,
    Field(discriminator="type"),
]
register_schema(VectorStoreChunkingStrategy, name="VectorStoreChunkingStrategy")


class SearchRankingOptions(BaseModel):
    """Options for ranking and filtering search results.

    This class configures how search results are ranked and filtered. You can use algorithm-based
    rerankers (weighted, RRF) or neural rerankers. Defaults from VectorStoresConfig are
    used when parameters are not provided.

    Examples:
        # Weighted ranker with custom alpha
        SearchRankingOptions(ranker="weighted", alpha=0.7)

        # RRF ranker with custom impact factor
        SearchRankingOptions(ranker="rrf", impact_factor=50.0)

        # Use config defaults (just specify ranker type)
        SearchRankingOptions(ranker="weighted")  # Uses alpha from VectorStoresConfig

        # Score threshold filtering
        SearchRankingOptions(ranker="weighted", score_threshold=0.5)

    :param ranker: (Optional) Name of the ranking algorithm to use. Supported values:
        - "weighted": Weighted combination of vector and keyword scores
        - "rrf": Reciprocal Rank Fusion algorithm
        - "neural": Neural reranking model (requires model parameter)
        Note: For OpenAI API compatibility, any string value is accepted, but only the above values are supported.
    :param score_threshold: (Optional) Minimum relevance score threshold for results. Default: 0.0
    :param alpha: (Optional) Weight factor for weighted ranker (0-1).
        - 0.0 = keyword only
        - 0.5 = equal weight (default)
        - 1.0 = vector only
        Only used when ranker="weighted" and weights is not provided.
        Falls back to VectorStoresConfig.chunk_retrieval_params.weighted_search_alpha if not provided.
    :param impact_factor: (Optional) Impact factor (k) for RRF algorithm.
        Lower values emphasize higher-ranked results. Default: 60.0 (optimal from research).
        Only used when ranker="rrf".
        Falls back to VectorStoresConfig.chunk_retrieval_params.rrf_impact_factor if not provided.
    :param weights: (Optional) Dictionary of weights for combining different signal types.
        Keys can be "vector", "keyword", "neural". Values should sum to 1.0.
        Used when combining algorithm-based reranking with neural reranking.
        Example: {"vector": 0.3, "keyword": 0.3, "neural": 0.4}
    :param model: (Optional) Model identifier for neural reranker (e.g., "transformers/Qwen/Qwen3-Reranker-0.6B").
        Required when ranker="neural" or when weights contains "neural".
    """

    ranker: str | None = None
    # NOTE: OpenAI File Search Tool requires threshold to be between 0 and 1, however
    # we don't guarantee that the score is between 0 and 1, so will leave this unconstrained
    # and let the provider handle it
    score_threshold: float | None = Field(default=0.0)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0, description="Weight factor for weighted ranker")
    impact_factor: float | None = Field(default=None, gt=0.0, description="Impact factor for RRF algorithm")
    weights: dict[str, float] | None = Field(
        default=None,
        description="Weights for combining vector, keyword, and neural scores. Keys: 'vector', 'keyword', 'neural'",
    )
    model: str | None = Field(default=None, description="Model identifier for neural reranker")

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return v
        allowed_keys = {"vector", "keyword", "neural"}
        if not all(key in allowed_keys for key in v.keys()):
            raise ValueError(f"weights keys must be from {allowed_keys}")
        if abs(sum(v.values()) - 1.0) > 0.001:
            raise ValueError("weights must sum to 1.0")
        return v


@json_schema_type
class VectorStoreFileLastError(BaseModel):
    """Error information for failed vector store file processing.

    :param code: Error code indicating the type of failure
    :param message: Human-readable error message describing the failure
    """

    code: Literal["server_error", "unsupported_file", "invalid_file"]
    message: str


VectorStoreFileStatus = Literal["in_progress", "completed", "cancelled", "failed"]
register_schema(VectorStoreFileStatus, name="VectorStoreFileStatus")


# VectorStoreFileAttributes type with OpenAPI constraints
VectorStoreFileAttributes = Annotated[
    dict[str, Annotated[str, Field(max_length=512)] | float | bool],
    Field(
        max_length=16,
        json_schema_extra={
            "propertyNames": {"type": "string", "maxLength": 64},
            "x-oaiTypeLabel": "map",
        },
        description=(
            "Set of 16 key-value pairs that can be attached to an object. This can be "
            "useful for storing additional information about the object in a structured "
            "format, and querying for objects via API or the dashboard. Keys are strings "
            "with a maximum length of 64 characters. Values are strings with a maximum "
            "length of 512 characters, booleans, or numbers."
        ),
    ),
]


def _sanitize_vector_store_attributes(metadata: dict[str, Any] | None) -> dict[str, str | float | bool]:
    """
    Sanitize metadata to VectorStoreFileAttributes spec (max 16 properties, primitives only).

    Converts dict[str, Any] to dict[str, str | float | bool]:
    - Preserves: str (truncated to 512 chars), bool, int/float (as float)
    - Converts: list -> comma-separated string
    - Filters: dict, None, other types
    - Enforces: max 16 properties, max 64 char keys, max 512 char string values
    """
    if not metadata:
        return {}

    sanitized: dict[str, str | float | bool] = {}
    for key, value in metadata.items():
        # Enforce max 16 properties
        if len(sanitized) >= 16:
            break

        # Enforce max 64 char keys
        if len(key) > 64:
            continue

        # Convert to supported primitive types
        if isinstance(value, bool):
            sanitized[key] = value
        elif isinstance(value, int | float):
            sanitized[key] = float(value)
        elif isinstance(value, str):
            # Enforce max 512 char string values
            sanitized[key] = value[:512] if len(value) > 512 else value
        elif isinstance(value, list):
            # Convert lists to comma-separated strings (max 512 chars)
            list_str = ", ".join(str(item) for item in value)
            sanitized[key] = list_str[:512] if len(list_str) > 512 else list_str

    return sanitized


@json_schema_type
class VectorStoreFileObject(BaseModel):
    """OpenAI Vector Store File object.

    :param id: Unique identifier for the file
    :param object: Object type identifier, always "vector_store.file"
    :param attributes: Key-value attributes associated with the file
    :param chunking_strategy: Strategy used for splitting the file into chunks
    :param created_at: Timestamp when the file was added to the vector store
    :param last_error: Error information if file processing failed, or null if no errors
    :param status: Current processing status of the file
    :param usage_bytes: Storage space used by this file in bytes
    :param vector_store_id: ID of the vector store containing this file
    """

    id: str
    object: Literal["vector_store.file"] = "vector_store.file"
    attributes: VectorStoreFileAttributes | None = None
    chunking_strategy: VectorStoreChunkingStrategy
    created_at: int
    last_error: VectorStoreFileLastError | None = None
    status: VectorStoreFileStatus
    usage_bytes: int = 0
    vector_store_id: str

    @field_validator("attributes", mode="before")
    @classmethod
    def _validate_attributes(cls, v: dict[str, Any] | None) -> dict[str, str | float | bool] | None:
        """Sanitize attributes to match VectorStoreFileAttributes OpenAPI spec."""
        if v is None:
            return None
        return _sanitize_vector_store_attributes(v)


@json_schema_type
class VectorStoreListFilesResponse(BaseModel):
    """Response from listing files in a vector store.

    :param object: Object type identifier, always "list"
    :param data: List of vector store file objects
    :param first_id: ID of the first file in the list for pagination
    :param last_id: ID of the last file in the list for pagination
    :param has_more: Whether there are more files available beyond this page
    """

    object: Literal["list"] = "list"
    data: list[VectorStoreFileObject]
    first_id: str
    last_id: str
    has_more: bool


@json_schema_type
class VectorStoreFileDeleteResponse(BaseModel):
    """Response from deleting a vector store file.

    :param id: Unique identifier of the deleted file
    :param object: Object type identifier for the deletion response
    :param deleted: Whether the deletion operation was successful
    """

    id: str
    object: Literal["vector_store.file.deleted"] = "vector_store.file.deleted"
    deleted: bool


@json_schema_type
class VectorStoreFileBatchObject(BaseModel):
    """OpenAI Vector Store File Batch object.

    :param id: Unique identifier for the file batch
    :param object: Object type identifier, always "vector_store.files_batch"
    :param created_at: Timestamp when the file batch was created
    :param vector_store_id: ID of the vector store containing the file batch
    :param status: Current processing status of the file batch
    :param file_counts: File processing status counts for the batch
    """

    id: str
    object: Literal["vector_store.files_batch"] = "vector_store.files_batch"
    created_at: int
    vector_store_id: str
    status: VectorStoreFileStatus
    file_counts: VectorStoreFileCounts


@json_schema_type
class VectorStoreFilesListInBatchResponse(BaseModel):
    """Response from listing files in a vector store file batch.

    :param object: Object type identifier, always "list"
    :param data: List of vector store file objects in the batch
    :param first_id: ID of the first file in the list for pagination
    :param last_id: ID of the last file in the list for pagination
    :param has_more: Whether there are more files available beyond this page
    """

    object: Literal["list"] = "list"
    data: list[VectorStoreFileObject]
    first_id: str
    last_id: str
    has_more: bool


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICreateVectorStoreRequestWithExtraBody(BaseModel, extra="allow"):
    """Request to create a vector store with extra_body support.

    :param name: (Optional) A name for the vector store
    :param description: (Optional) Description of the vector store
    :param file_ids: List of file IDs to include in the vector store
    :param expires_after: (Optional) Expiration policy for the vector store
    :param chunking_strategy: (Optional) Strategy for splitting files into chunks
    :param metadata: Set of key-value pairs that can be attached to the vector store
    """

    name: str | None = None
    description: str | None = None
    file_ids: list[str] | None = None
    expires_after: VectorStoreExpirationAfter | None = None
    chunking_strategy: VectorStoreChunkingStrategy | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class VectorStoreFileBatchFileEntry(BaseModel):
    """A file entry for creating a vector store file batch with per-file options.

    :param file_id: A File ID that the vector store should use
    :param chunking_strategy: (Optional) The chunking strategy used to chunk the file
    :param attributes: (Optional) Key-value attributes to store with the file
    """

    file_id: str
    chunking_strategy: VectorStoreChunkingStrategy | None = None
    attributes: VectorStoreFileAttributes | None = None


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICreateVectorStoreFileBatchRequestWithExtraBody(BaseModel, extra="allow"):
    """Request to create a vector store file batch with extra_body support.

    :param file_ids: A list of File IDs that the vector store should use. Mutually exclusive with files
    :param files: A list of file entries with per-file options. Mutually exclusive with file_ids
    :param attributes: (Optional) Key-value attributes to store with the files
    :param chunking_strategy: (Optional) The chunking strategy used to chunk the file(s). Defaults to auto
    """

    file_ids: list[str] = Field(default_factory=list)
    files: list[VectorStoreFileBatchFileEntry] | None = None
    attributes: VectorStoreFileAttributes | None = None
    chunking_strategy: VectorStoreChunkingStrategy | None = None


@json_schema_type
class ChunkForDeletion(BaseModel):
    """Information needed to delete a chunk from a vector store.

    :param chunk_id: The ID of the chunk to delete
    :param document_id: The ID of the document this chunk belongs to
    """

    chunk_id: str
    document_id: str


@json_schema_type
class InsertChunksRequest(BaseModel):
    """Request body for inserting chunks into a vector store."""

    vector_store_id: str = Field(description="The ID of the vector store to insert chunks into.")
    chunks: list[EmbeddedChunk] = Field(description="The list of embedded chunks to insert.")
    ttl_seconds: int | None = Field(default=None, description="Time-to-live in seconds for the inserted chunks.")


@json_schema_type
class DeleteChunksRequest(BaseModel):
    """Request body for deleting chunks from a vector store."""

    vector_store_id: str = Field(description="The ID of the vector store to delete chunks from.")
    chunks: list[ChunkForDeletion] = Field(description="The list of chunks to delete.")


@json_schema_type
class QueryChunksRequest(BaseModel):
    """Request body for querying chunks from a vector store."""

    vector_store_id: str = Field(description="The ID of the vector store to query.")
    query: InterleavedContent = Field(description="The query content to search for.")
    params: dict[str, Any] | None = Field(default=None, description="Additional query parameters.")


@json_schema_type
class OpenAIUpdateVectorStoreRequest(BaseModel):
    """Request body for updating a vector store."""

    name: str | None = Field(default=None, description="The new name for the vector store.")
    expires_after: VectorStoreExpirationAfter | None = Field(
        default=None, description="Expiration policy for the vector store."
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata to associate with the vector store.")


@json_schema_type
class OpenAISearchVectorStoreRequest(BaseModel):
    """Request body for searching a vector store."""

    query: str | list[str] = Field(description="The search query string or list of query strings.")
    filters: dict[str, Any] | None = Field(default=None, description="Filters to apply to the search.")
    max_num_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return.")
    ranking_options: SearchRankingOptions | None = Field(default=None, description="Options for ranking results.")
    rewrite_query: bool = Field(default=False, description="Whether to rewrite the query for better results.")
    search_mode: str | None = Field(default="vector", description="The search mode to use (e.g., 'vector', 'keyword').")


@json_schema_type
class OpenAIAttachFileRequest(BaseModel):
    """Request body for attaching a file to a vector store."""

    file_id: str = Field(description="The ID of the file to attach.")
    attributes: VectorStoreFileAttributes | None = Field(
        default=None,
        description="Attributes to associate with the file.",
    )
    chunking_strategy: VectorStoreChunkingStrategy | None = Field(
        default=None, description="Strategy for chunking the file content."
    )


@json_schema_type
class OpenAIUpdateVectorStoreFileRequest(BaseModel):
    """Request body for updating a vector store file."""

    attributes: dict[str, Any] = Field(description="The new attributes for the file.")


__all__ = [
    "Chunk",
    "ChunkForDeletion",
    "ChunkMetadata",
    "DEFAULT_CHUNK_OVERLAP_TOKENS",
    "DEFAULT_CHUNK_SIZE_TOKENS",
    "DeleteChunksRequest",
    "EmbeddedChunk",
    "InsertChunksRequest",
    "MAX_PAGINATION_LIMIT",
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
    "VectorStoreExpirationAfter",
    "VectorStoreFileAttributes",
    "VectorStoreFileBatchFileEntry",
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
    "VectorStoreStatus",
]
