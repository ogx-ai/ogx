# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import io
import threading
import time
import uuid
from typing import Any

from fastapi import UploadFile
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from ogx.log import get_logger
from ogx.providers.inline.file_processor.zip_utils import validate_zip_content
from ogx.providers.utils.files.response import response_body_bytes
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.vector_io import (
    Chunk,
    ChunkMetadata,
    VectorStoreChunkingStrategy,
)

from .config import UnstructuredFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")


class UnstructuredFileProcessor:
    """Local Unstructured file processor supporting 65+ formats.

    Uses the open-source Unstructured library for local document parsing.
    Supports PDF, DOCX, PPTX, XLSX, HTML, EML, MSG, audio transcription,
    and many other formats.

    WARNING: Table detection is unreliable in local mode (GitHub issue #2997).
    For production table extraction, use remote::unstructured-api instead.

    System dependencies required:
    - libmagic-dev (file type detection)
    - poppler-utils (PDF processing)
    - tesseract-ocr (OCR support)
    - libreoffice (optional, for Office document conversion)
    """

    def __init__(self, config: UnstructuredFileProcessorConfig, files_api=None) -> None:
        self.config = config
        self.files_api = files_api
        self._partition_lock = threading.Lock()

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using local Unstructured library."""
        file_id = request.file_id
        chunking_strategy = request.chunking_strategy

        # Validate input
        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        # Get file content
        if file:
            content = await file.read()
            filename = file.filename or f"{uuid.uuid4()}.bin"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = await response_body_bytes(content_response)

        # Process in thread pool (blocking library)
        return await asyncio.to_thread(self._process_content, content, filename, file_id, chunking_strategy, start_time)

    def _process_content(
        self,
        content: bytes,
        filename: str,
        file_id: str | None,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        start_time: float,
    ) -> ProcessFileResponse:
        """Partition and chunk file content. Runs in a thread."""
        validate_zip_content(content, filename)

        log.info(
            "Partitioning file with Unstructured",
            filename=filename,
            size_bytes=len(content),
            strategy=self.config.strategy,
        )

        file_like = io.BytesIO(content)

        with self._partition_lock:
            elements = partition(
                file=file_like,
                metadata_filename=filename,
                strategy=self.config.strategy,
                include_page_breaks=self.config.include_page_breaks,
                skip_infer_table_types=self.config.skip_infer_table_types,
                extract_images_in_pdf=self.config.extract_images_in_pdf,
                languages=self.config.languages,
            )

        log.info(
            "Unstructured partitioning complete",
            filename=filename,
            element_count=len(elements),
        )

        document_id = file_id if file_id else str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        # Create chunks from elements
        chunks = self._create_chunks(elements, document_id, chunking_strategy, document_metadata)

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_metadata: dict[str, Any] = {
            "processor": "unstructured",
            "processing_time_ms": processing_time_ms,
            "extraction_method": "unstructured-local",
            "file_size_bytes": len(content),
            "total_elements": len(elements),
            "strategy": self.config.strategy,
        }

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _create_chunks(
        self,
        elements: list[Any],  # List of Element objects from unstructured
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert Unstructured elements to OGX Chunks.

        Chunking semantics (matching remote::unstructured-api pattern):
        - chunking_strategy is None -> each element becomes one chunk
        - chunking_strategy.type == "auto" -> use chunk_by_title with configured defaults
        - chunking_strategy.type == "static" -> use chunk_by_title with provided max_tokens
        """
        if not elements:
            return []

        if not chunking_strategy:
            # No chunking - each element becomes one chunk
            return self._elements_to_individual_chunks(elements, document_id, document_metadata)

        # With chunking - use Unstructured's chunk_by_title (matches API behavior)
        # Determine max_characters based on strategy (same logic as remote API)
        if chunking_strategy.type == "auto":
            max_tokens = self.config.default_chunk_size_tokens
        elif chunking_strategy.type == "static":
            max_tokens = chunking_strategy.static.max_chunk_size_tokens
        else:
            max_tokens = self.config.default_chunk_size_tokens

        # Convert tokens to characters (same conversion as remote API: 1 token ≈ 4 characters)
        max_characters = max_tokens * 4

        log.info(
            "Chunking elements with chunk_by_title",
            max_tokens=max_tokens,
            max_characters=max_characters,
        )

        # Use Unstructured's native chunking
        chunked_elements = chunk_by_title(
            elements,
            max_characters=max_characters,
        )

        log.info(
            "Chunked elements",
            original_count=len(elements),
            chunk_count=len(chunked_elements),
        )

        return self._elements_to_individual_chunks(chunked_elements, document_id, document_metadata)

    def _elements_to_individual_chunks(
        self,
        elements: list[Any],
        document_id: str,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert element objects to OGX Chunk objects.

        Elements are Unstructured Element objects with:
        - .text attribute (str)
        - .category attribute (str) - element type
        - .metadata object with .to_dict() method
        """
        chunks: list[Chunk] = []

        for idx, element in enumerate(elements):
            # Extract text - Use .text attribute (not dict access)
            text = element.text

            # Skip empty elements
            if not text or not text.strip():
                continue

            # Get metadata - Use .metadata.to_dict() for serialization
            elem_metadata_dict = element.metadata.to_dict()
            page_number = elem_metadata_dict.get("page_number")
            element_type = element.category  # Use .category attribute

            # Generate chunk_id
            chunk_id = generate_chunk_id(document_id, text, str(idx))

            # Calculate token count (heuristic: 1 token ≈ 4 characters)
            content_token_count = len(text) // 4

            # Build metadata dict
            metadata_dict: dict[str, Any] = {
                "document_id": document_id,
                "element_type": element_type,
                "element_index": idx,
                **document_metadata,
            }
            if page_number is not None:
                metadata_dict["page_number"] = page_number

            # Include coordinates if available (useful for position-aware retrieval)
            if elem_metadata_dict.get("coordinates"):
                metadata_dict["coordinates"] = elem_metadata_dict["coordinates"]

            chunk = Chunk(
                content=text,
                chunk_id=chunk_id,
                metadata=metadata_dict,
                chunk_metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source=document_metadata.get("filename", ""),
                    content_token_count=content_token_count,
                ),
            )

            chunks.append(chunk)

        log.info(
            "Converted elements to chunks",
            total_elements=len(elements),
            total_chunks=len(chunks),
            skipped=len(elements) - len(chunks),
        )

        return chunks

    async def shutdown(self) -> None:
        """Shutdown hook for cleanup."""
        pass
