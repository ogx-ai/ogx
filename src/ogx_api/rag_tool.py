# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

from ogx_api.common.content_types import URL, InterleavedContent


class RRFRanker(BaseModel):
    """
    Reciprocal Rank Fusion (RRF) ranker configuration.

    :param type: The type of ranker, always "rrf"
    :param impact_factor: The impact factor for RRF scoring. Higher values give more weight to higher-ranked results.
                         Must be greater than 0
    """

    type: Literal["rrf"] = "rrf"
    impact_factor: float = Field(default=60.0, gt=0.0)  # default of 60 for optimal performance


class WeightedRanker(BaseModel):
    """
    Weighted ranker configuration that combines vector and keyword scores.

    :param type: The type of ranker, always "weighted"
    :param alpha: Weight factor between 0 and 1.
                 0 means only use keyword scores,
                 1 means only use vector scores,
                 values in between blend both scores.
    """

    type: Literal["weighted"] = "weighted"
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight factor between 0 and 1. 0 means only keyword scores, 1 means only vector scores.",
    )


Ranker = Annotated[
    RRFRanker | WeightedRanker,
    Field(discriminator="type"),
]


class RAGDocument(BaseModel):
    """
    A document to be used for document ingestion in the RAG Tool.

    :param document_id: The unique identifier for the document.
    :param content: The content of the document.
    :param mime_type: The MIME type of the document.
    :param metadata: Additional metadata for the document.
    """

    document_id: str
    content: InterleavedContent | URL
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryResult(BaseModel):
    """Result of a RAG query containing retrieved content and metadata.

    :param content: (Optional) The retrieved content from the query
    :param metadata: Additional metadata about the query result
    """

    content: InterleavedContent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQueryGenerator(Enum):
    """Types of query generators for RAG systems.

    :cvar default: Default query generator using simple text processing
    :cvar llm: LLM-based query generator for enhanced query understanding
    :cvar custom: Custom query generator implementation
    """

    default = "default"
    llm = "llm"
    custom = "custom"


class RAGSearchMode(StrEnum):
    """
    Search modes for RAG query retrieval:
    - VECTOR: Uses vector similarity search for semantic matching
    - KEYWORD: Uses keyword-based search for exact matching
    - HYBRID: Combines both vector and keyword search for better results
    """

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class DefaultRAGQueryGeneratorConfig(BaseModel):
    """Configuration for the default RAG query generator.

    :param type: Type of query generator, always 'default'
    :param separator: String separator used to join query terms
    """

    type: Literal["default"] = "default"
    separator: str = " "


class LLMRAGQueryGeneratorConfig(BaseModel):
    """Configuration for the LLM-based RAG query generator.

    :param type: Type of query generator, always 'llm'
    :param model: Name of the language model to use for query generation
    :param template: Template string for formatting the query generation prompt
    """

    type: Literal["llm"] = "llm"
    model: str
    template: str


RAGQueryGeneratorConfig = Annotated[
    DefaultRAGQueryGeneratorConfig | LLMRAGQueryGeneratorConfig,
    Field(discriminator="type"),
]


class RAGQueryConfig(BaseModel):
    """
    Configuration for the RAG query generation.

    :param query_generator_config: Configuration for the query generator.
    :param max_tokens_in_context: Maximum number of tokens in the context.
    :param max_chunks: Maximum number of chunks to retrieve.
    :param chunk_template: Template for formatting each retrieved chunk in the context.
        Available placeholders: {index} (1-based chunk ordinal), {chunk.content} (chunk content string), {metadata} (chunk metadata dict).
        Default: "Result {index}\\nContent: {chunk.content}\\nMetadata: {metadata}\\n"
    :param mode: Search mode for retrieval—either "vector", "keyword", or "hybrid". Default "vector".
    :param ranker: Configuration for the ranker to use in hybrid search. Defaults to RRF ranker.
    """

    # This config defines how a query is generated using the messages
    # for memory bank retrieval.
    query_generator_config: RAGQueryGeneratorConfig = Field(default=DefaultRAGQueryGeneratorConfig())
    max_tokens_in_context: int = 4096
    max_chunks: int = 5
    chunk_template: str = "Result {index}\nContent: {chunk.content}\nMetadata: {metadata}\n"
    mode: RAGSearchMode | None = RAGSearchMode.VECTOR
    ranker: Ranker | None = Field(default=None)  # Only used for hybrid mode

    @field_validator("chunk_template")
    def validate_chunk_template(cls, v: str) -> str:
        if "{chunk.content}" not in v:
            raise ValueError("chunk_template must contain {chunk.content}")
        if "{index}" not in v:
            raise ValueError("chunk_template must contain {index}")
        if len(v) == 0:
            raise ValueError("chunk_template must not be empty")
        return v
