# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import time
import uuid
from typing import Any

import httpx
from fastapi import UploadFile

from ogx.log import get_logger
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import Files, RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.vector_io import (
    Chunk,
    ChunkMetadata,
    VectorStoreChunkingStrategy,
)

from .config import DoclingServeFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")


class DoclingServeFileProcessor:
    """Remote file processor that delegates to a Docling Serve instance.

    Uses the Docling Serve REST API for layout-aware document conversion
    and chunking, supporting PDF, DOCX, PPTX, HTML, images, and more.
    """

    def __init__(self, config: DoclingServeFileProcessorConfig, files_api: Files) -> None:
        self.config = config
        self.files_api = files_api

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["X-Api-Key"] = self.config.api_key.get_secret_value()
        return headers

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file by sending it to Docling Serve and return chunks."""
        file_id = request.file_id
        chunking_strategy = request.chunking_strategy

        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        if file:
            content = await file.read()
            filename = file.filename or "upload"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = content_response.body

        document_id = file_id if file_id else str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        suffix = os.path.splitext(filename)[1] or ".bin"
        mime_type = _get_mime_type(suffix)

        if chunking_strategy:
            chunks = await self._convert_and_chunk(
                content, filename, mime_type, document_id, chunking_strategy, document_metadata
            )
        else:
            chunks = await self._convert_no_chunk(content, filename, mime_type, document_id, document_metadata)

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_metadata: dict[str, Any] = {
            "processor": "docling-serve",
            "processing_time_ms": processing_time_ms,
            "extraction_method": "docling-serve",
            "file_size_bytes": len(content),
        }

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    async def _convert_no_chunk(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        document_id: str,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert a file via Docling Serve without chunking and return a single chunk."""
        url = f"{self.config.base_url}/convert/file"
        headers = self._get_headers()

        options = {
            "to_formats": '["md"]',
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                url,
                files={"files": (filename, content, mime_type)},
                data=options,
                headers=headers,
            )
            response.raise_for_status()

        result = response.json()
        md_content = result.get("document", {}).get("md_content", "")

        if not md_content or not md_content.strip():
            return []

        chunk_id = generate_chunk_id(document_id, md_content)
        return [
            Chunk(
                content=md_content,
                chunk_id=chunk_id,
                metadata={
                    "document_id": document_id,
                    **document_metadata,
                },
                chunk_metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source=document_metadata.get("filename", ""),
                    content_token_count=len(md_content.split()),
                ),
            )
        ]

    async def _convert_and_chunk(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert and chunk a file via Docling Serve's hybrid chunker endpoint."""
        url = f"{self.config.base_url}/chunk/hybrid/file"
        headers = self._get_headers()

        if chunking_strategy.type == "auto":
            max_tokens = self.config.default_chunk_size_tokens
        elif chunking_strategy.type == "static":
            max_tokens = chunking_strategy.static.max_chunk_size_tokens
        else:
            max_tokens = self.config.default_chunk_size_tokens

        options: dict[str, str] = {
            "chunking_max_tokens": str(max_tokens),
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                url,
                files={"files": (filename, content, mime_type)},
                data=options,
                headers=headers,
            )
            response.raise_for_status()

        result = response.json()
        raw_chunks = result.get("chunks", [])

        if not raw_chunks:
            return []

        chunks: list[Chunk] = []
        for i, raw_chunk in enumerate(raw_chunks):
            text = raw_chunk.get("text", "")
            if not text or not text.strip():
                continue

            chunk_window = str(i)
            chunk_id = generate_chunk_id(document_id, text, chunk_window)

            meta: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }

            headings = raw_chunk.get("meta", {}).get("headings", None)
            if headings:
                meta["headings"] = headings

            chunks.append(
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata=meta,
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=len(text.split()),
                        chunk_window=chunk_window,
                    ),
                )
            )

        return chunks

    async def shutdown(self) -> None:
        pass


def _get_mime_type(suffix: str) -> str:
    """Map file extension to MIME type."""
    mime_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".html": "text/html",
        ".htm": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }
    return mime_types.get(suffix.lower(), "application/octet-stream")
