# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import io
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import chardet
import tiktoken
from pypdf import PdfReader

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_numpy: Any = None
_numpy_lock = threading.Lock()


def _get_numpy() -> Any:
    global _numpy
    if _numpy is not None:
        return _numpy
    with _numpy_lock:
        if _numpy is not None:
            return _numpy
        import numpy

        _numpy = numpy
        return _numpy


from ogx.core.datatypes import VectorStoresConfig
from ogx.log import get_logger
from ogx.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from ogx.providers.utils.vector_io.filters import Filter
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api import (
    Chunk,
    ChunkForDeletion,
    ChunkMetadata,
    EmbeddedChunk,
    Inference,
    InsertChunksRequest,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIEmbeddingsRequestWithExtraBody,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorStore,
)
from ogx_api.inference import RerankRequest

log = get_logger(name=__name__, category="providers::utils")


@cache
def _get_encoding(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tiktoken encoding '{name}'. "
            "In air-gapped or network-restricted environments, the encoding must be pre-cached "
            "at image build time. Set TIKTOKEN_CACHE_DIR to a directory containing the cached "
            f"encoding file, or ensure the container image was built with the encoding pre-cached. "
            f"Original error: {e}"
        ) from e


def validate_tiktoken_encoding(name: str = "cl100k_base") -> None:
    """Validate that the tiktoken encoding is available.

    Call this during provider initialization so a misconfigured environment
    fails fast with a clear operator-facing message to end users on their first vector store file operation.

    Raises:
        RuntimeError: if the encoding cannot be loaded (e.g. air-gapped env
            without a pre-cached encoding file).
    """
    _get_encoding(name)


# Constants for reranker types
RERANKER_TYPE_RRF = "rrf"
RERANKER_TYPE_WEIGHTED = "weighted"
RERANKER_TYPE_NORMALIZED = "normalized"


def parse_pdf(data: bytes) -> str:
    """Extract text content from PDF binary data.

    Args:
        data: raw PDF bytes

    Returns:
        Concatenated text from all pages
    """
    # For PDF and DOC/DOCX files, we can't reliably convert to string
    pdf_bytes = io.BytesIO(data)
    pdf_reader = PdfReader(pdf_bytes)
    return "\n".join([page.extract_text() for page in pdf_reader.pages])


def content_from_data_and_mime_type(data: bytes | str, mime_type: str | None, encoding: str | None = None) -> str:
    """Convert raw data to a string based on its MIME type.

    Args:
        data: raw bytes or string content
        mime_type: MIME type of the data
        encoding: optional character encoding override

    Returns:
        Extracted text content as a string
    """
    if isinstance(data, str):
        return data

    mime_category = mime_type.split("/")[0] if mime_type else None
    if mime_category == "text":
        if not encoding:
            detected = chardet.detect(data)
            encoding = detected["encoding"] or "utf-8"

        # For text-based files (including CSV, MD)
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as e:
            log.warning(f"Decoding with encoding {encoding} failed: {e}")
            if encoding.lower() != "utf-8":
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError as e_utf8:
                    log.warning(f"Decoding with UTF-8 fallback also failed: {e_utf8}")
            raise e

    elif mime_type == "application/pdf":
        return parse_pdf(data)

    else:
        log.error("Could not extract content from data_url properly.")
        return ""


def make_overlapped_chunks(
    document_id: str,
    text: str,
    window_len: int,
    overlap_len: int,
    metadata: dict[str, Any],
    chunk_tokenizer_encoding: str = "cl100k_base",
) -> list[Chunk]:
    """Split `text` into overlapping windows while tracking the tokenizer used.

    The function converts the document and metadata into tokens via `tiktoken`,
    defaulting to `cl100k_base` but allowing callers to override the encoding name
    if they already know the embedding model's tokenizer. Each window advances by
    `window_len - overlap_len`, decodes for storage, and records the tokenizer name and token
    counts on both metadata layers so downstream consumers can stay within their limits.
    Downstream tokenizers may split the decoded text into more tokens than the original
    encoding, so adjust `window_len` when targeting a model whose tokenizer expands the
    chunk beyond its token limit."""
    encoding_name = chunk_tokenizer_encoding
    encoding = _get_encoding(encoding_name)
    chunk_tokenizer_name = f"tiktoken:{encoding_name}"
    tokens = encoding.encode(text)
    try:
        metadata_string = str(metadata)
    except Exception as e:
        raise ValueError("Failed to serialize metadata to string") from e

    metadata_tokens = encoding.encode(metadata_string)

    chunks = []
    for i in range(0, len(tokens), window_len - overlap_len):
        toks = tokens[i : i + window_len]
        chunk = encoding.decode(toks)
        chunk_window = f"{i}-{i + len(toks)}"
        chunk_id = generate_chunk_id(chunk, text, chunk_window)
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["document_id"] = document_id
        chunk_metadata["token_count"] = len(toks)
        chunk_metadata["metadata_token_count"] = len(metadata_tokens)
        chunk_metadata["chunk_tokenizer"] = chunk_tokenizer_name

        backend_chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            source=metadata.get("source", None),
            created_timestamp=metadata.get("created_timestamp", int(time.time())),
            updated_timestamp=int(time.time()),
            chunk_window=chunk_window,
            chunk_tokenizer=chunk_tokenizer_name,
            content_token_count=len(toks),
            metadata_token_count=len(metadata_tokens),
        )

        # chunk is a string
        chunks.append(
            Chunk(
                content=chunk,
                chunk_id=chunk_id,
                metadata=chunk_metadata,
                chunk_metadata=backend_chunk_metadata,
            )
        )

    return chunks


type EmbeddingSequence = Sequence[float | int | np.number] | NDArray[Any]


def _validate_embedding(embedding: EmbeddingSequence, index: int, expected_dimension: int):
    """Helper method to validate embedding format and dimensions"""
    np = _get_numpy()
    if not isinstance(embedding, (list | np.ndarray)):
        raise ValueError(f"Embedding at index {index} must be a list or numpy array, got {type(embedding)}")

    if isinstance(embedding, np.ndarray):
        if not np.issubdtype(embedding.dtype, np.number):
            raise ValueError(f"Embedding at index {index} contains non-numeric values")
    else:
        if not all(isinstance(e, (float | int | np.number)) for e in embedding):
            raise ValueError(f"Embedding at index {index} contains non-numeric values")

    if len(embedding) != expected_dimension:
        raise ValueError(f"Embedding at index {index} has dimension {len(embedding)}, expected {expected_dimension}")


class EmbeddingIndex(ABC):
    """Abstract base class for vector embedding storage and retrieval backends."""

    @abstractmethod
    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk]):
        raise NotImplementedError()

    @abstractmethod
    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]):
        raise NotImplementedError()

    @abstractmethod
    async def query_vector(
        self, embedding: "NDArray", k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def query_hybrid(
        self,
        embedding: "NDArray",
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def delete(self):
        raise NotImplementedError()


@dataclass
class VectorStoreWithIndex:
    """Associates a VectorStore with its EmbeddingIndex and inference API for chunk operations."""

    vector_store: VectorStore
    index: EmbeddingIndex
    inference_api: Inference
    vector_stores_config: VectorStoresConfig | None = None

    async def insert_chunks(
        self,
        request: InsertChunksRequest,
    ) -> None:
        # Validate embedding dimensions match the vector store
        for i, embedded_chunk in enumerate(request.chunks):
            _validate_embedding(embedded_chunk.embedding, i, self.vector_store.embedding_dimension)

        await self.index.add_chunks(request.chunks)

    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        config = self.vector_stores_config or VectorStoresConfig()

        params = request.params
        if params is None:
            params = {}
        k = params.get("max_chunks", 3)
        desired_max_num_results = params.get("max_num_results", 2)
        mode = params.get("mode")
        score_threshold = params.get("score_threshold", 0.0)

        # Extract filters from params (processed by router)
        filters = params.get("filters")

        # Get reranker configuration from params (set by openai_vector_store_mixin)
        # NOTE: Breaking change - removed support for old nested "ranker" format.
        #       Now uses flattened format: reranker_type and reranker_params.
        reranker_type = params.get("reranker_type")
        reranker_params = params.get("reranker_params", {})
        neural_reranking_enabled = False

        # If no ranker specified, use VectorStoresConfig default
        if reranker_type is None:
            reranker_type = (
                RERANKER_TYPE_RRF
                if config.chunk_retrieval_params.default_reranker_strategy == "rrf"
                else config.chunk_retrieval_params.default_reranker_strategy
            )
            reranker_params = {"impact_factor": config.chunk_retrieval_params.rrf_impact_factor}

        # Normalize reranker_type to use constants
        if reranker_type == "weighted":
            reranker_type = RERANKER_TYPE_WEIGHTED
            # Ensure alpha is set (use default if not provided)
            if "alpha" not in reranker_params:
                reranker_params["alpha"] = config.chunk_retrieval_params.weighted_search_alpha
        elif reranker_type == "rrf":
            reranker_type = RERANKER_TYPE_RRF
            # Ensure impact_factor is set (use default if not provided)
            if "impact_factor" not in reranker_params:
                reranker_params["impact_factor"] = config.chunk_retrieval_params.rrf_impact_factor
        elif reranker_type == "neural":
            # Neural reranking is being applied after initial retrieval
            neural_reranking_enabled = True
        elif reranker_type == "normalized":
            reranker_type = RERANKER_TYPE_NORMALIZED
        else:
            # Default to RRF for unknown strategies
            reranker_type = RERANKER_TYPE_RRF
            if "impact_factor" not in reranker_params:
                reranker_params["impact_factor"] = config.chunk_retrieval_params.rrf_impact_factor

        # Store neural model and weights from params if provided
        if "neural_model" in params:
            reranker_params["neural_model"] = params["neural_model"]
        if "neural_weights" in params:
            reranker_params["neural_weights"] = params["neural_weights"]

        query_string = interleaved_content_as_str(request.query)
        log.info(f"query_chunks(): query={query_string!r}, mode={mode}, k={k}, reranker_type={reranker_type}")

        if mode == "keyword":
            response = await self.index.query_keyword(query_string, k, score_threshold, filters)

        else:
            if "embedding_dimensions" in params:
                embeddings_request = OpenAIEmbeddingsRequestWithExtraBody(
                    model=self.vector_store.embedding_model,
                    input=[query_string],
                    dimensions=params.get("embedding_dimensions"),
                )
            else:
                embeddings_request = OpenAIEmbeddingsRequestWithExtraBody(
                    model=self.vector_store.embedding_model, input=[query_string]
                )
            embeddings_response = await self.inference_api.openai_embeddings(embeddings_request)
            np = _get_numpy()
            query_vector = np.array(embeddings_response.data[0].embedding, dtype=np.float32)
            if mode == "hybrid":
                response = await self.index.query_hybrid(
                    query_vector, query_string, k, score_threshold, reranker_type, reranker_params, filters
                )
            else:
                response = await self.index.query_vector(query_vector, k, score_threshold, filters)

        log.info(f"query_chunks(): retrieved {len(response.chunks)} chunks before neural reranking")
        for i, (chunk, score) in enumerate(zip(response.chunks, response.scores, strict=False)):
            preview = chunk.content[:120] if isinstance(chunk.content, str) else str(chunk.content)[:120]
            log.info(
                f"Chunk {i}: score={score:.4f} doc_id={chunk.metadata.get('document_id', 'N/A')} content={preview!r}"
            )

        # Apply neural reranking if enabled
        if neural_reranking_enabled and response.chunks:
            response = await self.apply_neural_rerank(query_string, response, desired_max_num_results, reranker_params)

        return response

    async def apply_neural_rerank(
        self,
        query_string: str,
        response: QueryChunksResponse,
        desired_max_num_results: int,
        reranker_params: dict[str, Any],
    ) -> QueryChunksResponse:
        """
        Rerank retrieved chunks using a neural reranker model via the inference API.
        """
        reranker_model = reranker_params.get("model")

        if not reranker_model and self.vector_stores_config and self.vector_stores_config.default_reranker_model:
            config = self.vector_stores_config.default_reranker_model
            reranker_model = f"{config.provider_id}/{config.model_id}"

        if not reranker_model:
            log.warning(
                "Neural reranking requested but no reranker model configured. Returning results without reranking."
            )
            return response

        # Extract text contents from chunks for reranking
        text_from_chunks: list[
            str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam
        ] = []
        for chunk in response.chunks:
            if isinstance(chunk.content, str):
                text_from_chunks.append(chunk.content)
            else:
                text_from_chunks.append(interleaved_content_as_str(chunk.content))

        try:
            rerank_response = await self.inference_api.rerank(
                RerankRequest(
                    model=reranker_model,
                    query=query_string,
                    items=text_from_chunks,
                    max_num_results=desired_max_num_results,
                )
            )

        except Exception as e:
            log.error(f"Neural reranking failed: {e}. Returning original results.")
            return response

        log.info(f"Rerank Response: {rerank_response.data}")

        # Reorder chunks and scores based on neural rerank results
        reranked_chunks = []
        reranked_scores = []
        for reranked_chunk in rerank_response.data:
            if reranked_chunk.index < len(response.chunks):
                reranked_chunks.append(response.chunks[reranked_chunk.index])
                reranked_scores.append(reranked_chunk.relevance_score)

        log.info(f"Neural rerank: reranked {len(reranked_chunks)} chunks using model={reranker_model}")
        for i, (chunk, score) in enumerate(zip(reranked_chunks, reranked_scores, strict=False)):
            preview = chunk.content[:120] if isinstance(chunk.content, str) else str(chunk.content)[:120]
            log.info(
                f"Chunk {i}: relevance_score={score:.4f} doc_id={chunk.metadata.get('document_id', 'N/A')} content={preview!r}"
            )

        return QueryChunksResponse(chunks=reranked_chunks, scores=reranked_scores)

    # Note: File processing for vector stores now happens at the
    # openai_attach_file_to_vector_store level using file_id.
    # This VectorStoreWithIndex class focuses on chunk operations.
