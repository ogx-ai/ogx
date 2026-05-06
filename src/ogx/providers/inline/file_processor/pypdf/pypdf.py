# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import mimetypes
import time
import uuid
from typing import Any

import chardet
from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from ogx.log import get_logger
from ogx.providers.utils.memory.vector_store import make_overlapped_chunks
from ogx_api.file_processors import ProcessFileResponse
from ogx_api.files import RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.vector_io import (
    Chunk,
    VectorStoreChunkingStrategy,
)

from .config import PyPDFFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

# Window size for single-chunk mode (no chunking strategy)
# Large enough to fit any reasonable document in one chunk
SINGLE_CHUNK_WINDOW_TOKENS = 1_000_000


class PyPDFFileProcessor:
    """PyPDF-based file processor for PDF documents."""

    def __init__(self, config: PyPDFFileProcessorConfig, files_api) -> None:
        self.config = config
        self.files_api = files_api

    async def process_file(
        self,
        file: UploadFile | None = None,
        file_id: str | None = None,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> ProcessFileResponse:
        """Process a file and return chunks. Supports PDF and text-based files."""

        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        # Upload size limits are enforced by the router layer (upload_safety.py).
        # The provider trusts that `file` has already been bounded-read and
        # `file_id` references a file accepted by the Files API.
        if file:
            content = await file.read()
            filename = file.filename or f"{uuid.uuid4()}.pdf"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = content_response.body

        mime_type, _ = mimetypes.guess_type(filename)
        mime_category = mime_type.split("/")[0] if (mime_type and "/" in mime_type) else None

        if mime_type == "application/pdf":
            return self._process_pdf(content, filename, file_id, chunking_strategy, start_time)
        elif mime_category == "text":
            return self._process_text(content, filename, file_id, chunking_strategy, start_time)
        else:
            # Attempt text decoding as a fallback for unknown types
            log.warning("Unknown mime type, attempting text extraction", mime_type=mime_type, filename=filename)
            return self._process_text(content, filename, file_id, chunking_strategy, start_time)

    def _process_pdf(
        self,
        content: bytes,
        filename: str,
        file_id: str | None,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        start_time: float,
    ) -> ProcessFileResponse:
        """Process a PDF file."""
        pdf_bytes = io.BytesIO(content)
        reader = PdfReader(pdf_bytes)

        if reader.is_encrypted:
            raise HTTPException(status_code=422, detail="Password-protected PDFs are not supported")

        text_content, failed_pages = self._extract_pdf_text(reader)

        if self.config.clean_text:
            text_content = self._clean_text(text_content)

        pdf_metadata = {}
        if self.config.extract_metadata:
            pdf_metadata = self._extract_pdf_metadata(reader)

        document_id = str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename, **pdf_metadata}
        if file_id:
            document_metadata["file_id"] = file_id

        processing_time_ms = int((time.time() - start_time) * 1000)
        response_metadata: dict[str, Any] = {
            "processor": "pypdf",
            "processing_time_ms": processing_time_ms,
            "page_count": pdf_metadata.get("page_count", 0),
            "extraction_method": "pypdf",
            "file_size_bytes": len(content),
            **pdf_metadata,
        }
        if failed_pages:
            response_metadata["failed_pages"] = failed_pages

        if not text_content or not text_content.strip():
            return ProcessFileResponse(chunks=[], metadata=response_metadata)

        chunks = self._create_chunks(text_content, document_id, chunking_strategy, document_metadata)
        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _process_text(
        self,
        content: bytes,
        filename: str,
        file_id: str | None,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        start_time: float,
    ) -> ProcessFileResponse:
        """Process a text-based file (txt, csv, md, etc.)."""
        detected = chardet.detect(content)
        encoding = detected["encoding"] or "utf-8"
        try:
            text_content = content.decode(encoding)
        except UnicodeDecodeError:
            text_content = content.decode("utf-8", errors="replace")

        if self.config.clean_text:
            text_content = self._clean_text(text_content)

        document_id = str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        processing_time_ms = int((time.time() - start_time) * 1000)
        response_metadata: dict[str, Any] = {
            "processor": "text",
            "processing_time_ms": processing_time_ms,
            "extraction_method": "text",
            "file_size_bytes": len(content),
        }

        if not text_content or not text_content.strip():
            return ProcessFileResponse(chunks=[], metadata=response_metadata)

        chunks = self._create_chunks(text_content, document_id, chunking_strategy, document_metadata)
        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _extract_pdf_text(self, reader: PdfReader) -> tuple[str, list[str]]:
        """Extract text from all pages of a parsed PDF."""
        # Extract text from all pages
        text_parts = []
        failed_pages = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
            except Exception as e:
                failed_pages.append(f"page {page_num + 1}: {e}")
                continue
            if page_text:
                if not self.config.clean_text or page_text.strip():
                    text_parts.append(page_text)

        return "\n".join(text_parts), failed_pages

    def _extract_pdf_metadata(self, reader: PdfReader) -> dict[str, Any]:
        """Extract metadata from a parsed PDF."""
        metadata: dict[str, Any] = {"page_count": len(reader.pages)}

        if reader.metadata:
            pdf_metadata = reader.metadata

            keys = [
                "title",
                "author",
                "subject",
                "creator",
                "producer",
                "creation_date",
                "modification_date",
            ]

            for key in keys:
                value = getattr(pdf_metadata, key, None)
                if value:
                    metadata[key] = str(value)

        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Strip whitespace and normalize
            cleaned_line = " ".join(line.split())
            if cleaned_line:  # Only keep non-empty lines
                cleaned_lines.append(cleaned_line)

        return "\n".join(cleaned_lines)

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Create chunks from text content using make_overlapped_chunks.

        Chunking semantics:
        - chunking_strategy is None → return single chunk (large window, no overlap)
        - chunking_strategy.type == "auto" → use configured defaults
        - chunking_strategy.type == "static" → use provided values
        """
        # Determine chunk parameters based on strategy
        if not chunking_strategy:
            # No chunking - use very large window to get single chunk
            chunk_size = SINGLE_CHUNK_WINDOW_TOKENS
            overlap_size = 0
        elif chunking_strategy.type == "auto":
            # Use configured defaults for auto chunking
            chunk_size = self.config.default_chunk_size_tokens
            overlap_size = self.config.default_chunk_overlap_tokens
        elif chunking_strategy.type == "static":
            chunk_size = chunking_strategy.static.max_chunk_size_tokens
            overlap_size = chunking_strategy.static.chunk_overlap_tokens
        elif chunking_strategy.type == "contextual":
            chunk_size = chunking_strategy.contextual.max_chunk_size_tokens
            overlap_size = chunking_strategy.contextual.chunk_overlap_tokens
        else:
            chunk_size = self.config.default_chunk_size_tokens
            overlap_size = self.config.default_chunk_overlap_tokens

        # Prepare metadata for chunks (include filename and file_id)
        chunks_metadata_dict: dict[str, Any] = {
            "document_id": document_id,
            **document_metadata,
        }

        # Create chunks using existing utility (returns Chunk objects directly)
        return make_overlapped_chunks(
            document_id=document_id,
            text=text,
            window_len=chunk_size,
            overlap_len=overlap_size,
            metadata=chunks_metadata_dict,
        )
