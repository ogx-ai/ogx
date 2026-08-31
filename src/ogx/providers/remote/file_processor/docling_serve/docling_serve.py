# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import time
import uuid
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import httpx
from docling.datamodel.base_models import OutputFormat
from docling.datamodel.service.chunking import HybridChunkerOptions
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.responses import ChunkedDocumentResultItem
from docling.datamodel.service.targets import ZipTarget
from docling.service_client import AsyncDoclingServiceClient, RawServiceResult
from docling_core.types.io import DocumentStream
from fastapi import UploadFile

from ogx.log import get_logger
from ogx.providers.utils.files.response import response_body_bytes
from ogx.providers.utils.files.structural_metadata import structural_metadata_as_attributes
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api.common.errors import InvalidParameterError
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import Files, RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.vector_io import (
    Chunk,
    ChunkMetadata,
    VectorStoreChunkingStrategy,
)

from .config import DoclingServeFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

DOCLING_SERVE_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "text/html",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
}


class DoclingServeFileProcessor:
    """Remote file processor that delegates to a Docling Serve instance.

    Uses the Docling Serve REST API for layout-aware document conversion
    and chunking, supporting PDF, DOCX, PPTX, HTML, images, and more.
    """

    def __init__(self, config: DoclingServeFileProcessorConfig, files_api: Files) -> None:
        self.config = config
        self.files_api = files_api
        # Normalize base_url: AsyncDoclingServiceClient rejects URLs ending with /v1
        # Strip /v1 suffix for backward compatibility with old configs
        normalized_url = config.base_url.rstrip("/")
        if normalized_url.endswith("/v1"):
            normalized_url = normalized_url.removesuffix("/v1")
        self.config.base_url = normalized_url

    def supported_mime_types(self) -> set[str] | None:
        return DOCLING_SERVE_MIME_TYPES

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["X-Api-Key"] = self.config.api_key.get_secret_value()
        return headers

    def _get_max_tokens(self, chunking_strategy: VectorStoreChunkingStrategy) -> int:
        if chunking_strategy.type == "static":
            return chunking_strategy.static.max_chunk_size_tokens
        return self.config.default_chunk_size_tokens

    def _get_chunking_options(
        self,
        chunking_strategy: VectorStoreChunkingStrategy,
        request_options: dict[str, Any] | None,
    ) -> HybridChunkerOptions:
        use_markdown_tables = (request_options or {}).get("use_markdown_tables", False)
        if not isinstance(use_markdown_tables, bool):
            raise InvalidParameterError(
                param_name="options.use_markdown_tables",
                value=use_markdown_tables,
                constraint="Must be a boolean.",
            )

        return HybridChunkerOptions(
            max_tokens=self._get_max_tokens(chunking_strategy),
            use_markdown_tables=use_markdown_tables,
        )

    def _chunks_from_archive(
        self,
        content: bytes,
        document_id: str,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        with ZipFile(BytesIO(content)) as archive:
            chunk_files = [name for name in archive.namelist() if name.endswith(".chunks.jsonl")]
            if not chunk_files:
                raise ValueError("Failed to read Docling chunks: the response archive contains no chunks.jsonl file")

            raw_chunks = [
                ChunkedDocumentResultItem.model_validate_json(line)
                for chunk_file in chunk_files
                for line in archive.read(chunk_file).decode("utf-8").splitlines()
                if line.strip()
            ]

        chunks: list[Chunk] = []
        for raw_chunk in raw_chunks:
            text = raw_chunk.text
            if not text.strip():
                continue

            chunk_window = str(raw_chunk.chunk_index)
            chunk_id = generate_chunk_id(document_id, text, chunk_window)
            metadata: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }
            metadata.update(
                structural_metadata_as_attributes(
                    headings=raw_chunk.headings,
                    page_numbers=raw_chunk.page_numbers,
                )
            )

            chunks.append(
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=(
                            raw_chunk.num_tokens if raw_chunk.num_tokens is not None else len(text.split())
                        ),
                        chunk_window=chunk_window,
                    ),
                )
            )

        return chunks

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
            content = await response_body_bytes(content_response)

        document_id = file_id if file_id else str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        suffix = os.path.splitext(filename)[1] or ".bin"
        mime_type = _get_mime_type(suffix)

        # Try AsyncDoclingServiceClient first (async endpoints with WebSocket)
        chunks = None
        conversion_method = None

        if self.config.mode in ("async", "auto"):
            try:
                log.info(
                    "Converting with async endpoints using AsyncDoclingServiceClient from docling-slim",
                    mode=self.config.mode,
                    sdk_version="docling-slim>=2.117.0",
                )
                if chunking_strategy:
                    chunks = await self._convert_and_chunk_async(
                        content,
                        filename,
                        mime_type,
                        document_id,
                        chunking_strategy,
                        document_metadata,
                        request.options,
                    )
                else:
                    chunks = await self._convert_no_chunk_async(
                        content, filename, mime_type, document_id, document_metadata
                    )
                log.info(
                    "Successfully converted with async endpoints using AsyncDoclingServiceClient",
                    client_class="AsyncDoclingServiceClient",
                    sdk_module="docling.service_client",
                )
                conversion_method = "async"
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if self.config.mode == "auto":
                    log.warning("Async failed, falling back to sync", error=str(e))
                    chunks = None
                else:
                    raise

        # Fallback to sync endpoints if async failed or mode is sync
        if chunks is None:
            log.info("Using sync endpoints", mode=self.config.mode)
            if chunking_strategy:
                chunks = await self._convert_and_chunk(
                    content,
                    filename,
                    mime_type,
                    document_id,
                    chunking_strategy,
                    document_metadata,
                    request.options,
                )
            else:
                chunks = await self._convert_no_chunk(content, filename, mime_type, document_id, document_metadata)
            conversion_method = "sync"

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_metadata: dict[str, Any] = {
            "processor": "docling-serve",
            "processing_time_ms": processing_time_ms,
            "extraction_method": "docling-serve",
            "file_size_bytes": len(content),
            "conversion_method": conversion_method,
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
        url = f"{self.config.base_url}/v1/convert/file"
        headers = self._get_headers()

        options = {
            "to_formats": ["md"],
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

    async def _convert_no_chunk_async(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        document_id: str,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert file using async endpoints with AsyncDoclingServiceClient."""
        source = DocumentStream(name=filename, stream=BytesIO(content))
        async with AsyncDoclingServiceClient(
            url=self.config.base_url,
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else "",
            job_timeout=300.0,
        ) as client:
            job = await client.submit(
                source=source,
                options=ConvertDocumentsOptions(to_formats=[OutputFormat.MARKDOWN]),
            )
            result = await job.result()

        # Handle both local docling-serve (ConversionResult with .document)
        # and IBM SaaS (PresignedUrlConvertResponse with .documents and presigned URLs)
        md_content = ""
        if hasattr(result, "documents"):
            # IBM SaaS: PresignedUrlConvertResponse with presigned URLs
            if result.documents and result.documents[0].artifacts:
                artifact = result.documents[0].artifacts[0]
                # Download markdown from presigned URL
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.get(str(artifact.uri))
                    response.raise_for_status()
                    md_content = response.text
        elif hasattr(result, "document"):
            # Local docling-serve: ConversionResult with direct document
            md_content = result.document.export_to_markdown() if result.document else ""

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
        request_options: dict[str, Any] | None,
    ) -> list[Chunk]:
        """Convert and chunk a file via Docling Serve's synchronous convert endpoint."""
        url = f"{self.config.base_url}/v1/convert/file"
        headers = self._get_headers()
        chunking_options = self._get_chunking_options(chunking_strategy, request_options)
        options = {
            "to_formats": [OutputFormat.CHUNKS.value],
            "chunking_options": chunking_options.model_dump_json(exclude_none=True),
            "target_type": "zip",
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    url,
                    files={"files": (filename, content, mime_type)},
                    data=options,
                    headers=headers,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Chunk output is unavailable on older Docling Serve releases.
                if e.response.status_code in (404, 405):
                    raise InvalidParameterError(
                        param_name="chunking_strategy",
                        value=chunking_strategy.model_dump() if chunking_strategy else None,
                        constraint=(
                            "Chunk output through the convert endpoint is not supported by this Docling instance. "
                            "Either remove 'chunking_strategy' from your request or upgrade Docling Serve."
                        ),
                    ) from e
                raise

        return self._chunks_from_archive(response.content, document_id, document_metadata)

    async def _convert_and_chunk_async(
        self,
        content: bytes,
        filename: str,
        mime_type: str,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy,
        document_metadata: dict[str, Any],
        request_options: dict[str, Any] | None,
    ) -> list[Chunk]:
        """Convert and chunk a file via Docling Serve's asynchronous convert endpoint."""
        source = DocumentStream(name=filename, stream=BytesIO(content))
        async with AsyncDoclingServiceClient(
            url=self.config.base_url,
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else "",
            job_timeout=300.0,
        ) as client:
            try:
                job = await client.submit(
                    source=source,
                    options=ConvertDocumentsOptions(
                        to_formats=[OutputFormat.CHUNKS],
                        chunking_options=self._get_chunking_options(chunking_strategy, request_options),
                    ),
                    target=ZipTarget(),
                )
                response = await job.result()
            except httpx.HTTPStatusError as e:
                # Chunk output is unavailable on older Docling Serve releases.
                if e.response.status_code in (404, 405):
                    raise InvalidParameterError(
                        param_name="chunking_strategy",
                        value=chunking_strategy.model_dump() if chunking_strategy else None,
                        constraint=(
                            "Chunk output through the convert endpoint is not supported by this Docling instance. "
                            "Either remove 'chunking_strategy' from your request or upgrade Docling Serve."
                        ),
                    ) from e
                raise

        if not isinstance(response, RawServiceResult):
            raise ValueError("Failed to read Docling chunks: the convert endpoint did not return a ZIP archive")
        return self._chunks_from_archive(response.content, document_id, document_metadata)

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
