# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
import time
import uuid
from typing import Any

from fastapi import HTTPException, UploadFile
from markitdown import MarkItDown

from ogx.log import get_logger
from ogx.providers.utils.memory.vector_store import make_overlapped_chunks
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.vector_io import (
    Chunk,
    VectorStoreChunkingStrategy,
)

from .config import MarkItDownFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

SINGLE_CHUNK_WINDOW_TOKENS = 1_000_000


class MarkItDownFileProcessor:
    """MarkItDown-based file processor using Microsoft's MarkItDown library.

    Converts documents to Markdown and chunks for vector store ingestion.
    Supports PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, and code files.
    """

    def __init__(self, config: MarkItDownFileProcessorConfig, files_api) -> None:
        self.config = config
        self.files_api = files_api
        self.converter = MarkItDown()

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using MarkItDown and return chunks."""
        file_id = request.file_id
        chunking_strategy = request.chunking_strategy

        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        if file:
            content = await file.read()
            filename = file.filename or f"{uuid.uuid4()}.bin"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = content_response.body

        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()

            try:
                result = self.converter.convert(tmp.name)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to process file '{filename}': {e}",
                ) from e

        markdown_text = result.text_content or ""

        if not markdown_text.strip():
            processing_time_ms = int((time.time() - start_time) * 1000)
            return ProcessFileResponse(
                chunks=[],
                metadata={
                    "processor": "markitdown",
                    "processing_time_ms": processing_time_ms,
                    "extraction_method": "markitdown",
                    "file_size_bytes": len(content),
                },
            )

        document_id = str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        chunks = self._create_chunks(markdown_text, document_id, chunking_strategy, document_metadata)

        processing_time_ms = int((time.time() - start_time) * 1000)
        response_metadata: dict[str, Any] = {
            "processor": "markitdown",
            "processing_time_ms": processing_time_ms,
            "extraction_method": "markitdown",
            "file_size_bytes": len(content),
        }

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Create chunks from text content using make_overlapped_chunks."""
        if not chunking_strategy:
            chunk_size = SINGLE_CHUNK_WINDOW_TOKENS
            overlap_size = 0
        elif chunking_strategy.type == "auto":
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

        chunks_metadata_dict: dict[str, Any] = {
            "document_id": document_id,
            **document_metadata,
        }

        return make_overlapped_chunks(
            document_id=document_id,
            text=text,
            window_len=chunk_size,
            overlap_len=overlap_size,
            metadata=chunks_metadata_dict,
        )

    async def shutdown(self) -> None:
        pass
