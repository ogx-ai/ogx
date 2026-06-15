# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from typing import Any

from fastapi import UploadFile

from ogx.log import get_logger
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import Files, RetrieveFileContentRequest, RetrieveFileRequest
from ogx_api.vector_io import (
    Chunk,
    ChunkMetadata,
)

from .config import UnstructuredApiFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")


class UnstructuredApiFileProcessor:
    """Remote file processor that uses Unstructured.io SaaS API.

    Supports 65+ file formats including PDF, DOCX, PPTX, XLSX, EML, MSG, HTML,
    Markdown, and more. Uses the Unstructured.io cloud service for document
    parsing with advanced table and image detection.
    """

    def __init__(self, config: UnstructuredApiFileProcessorConfig, files_api: Files) -> None:
        self.config = config
        self.files_api = files_api

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using Unstructured.io API and return chunks."""
        file_id = request.file_id

        if not file and not file_id:
            raise ValueError("Either file or file_id must be provided")
        if file and file_id:
            raise ValueError("Cannot provide both file and file_id")

        start_time = time.time()

        # Get file content and metadata
        if file:
            content = await file.read()
            filename = file.filename or "upload"
        elif file_id:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=file_id))
            filename = file_info.filename

            content_response = await self.files_api.openai_retrieve_file_content(
                RetrieveFileContentRequest(file_id=file_id)
            )
            content = bytes(content_response.body)

        document_id = file_id if file_id else str(uuid.uuid4())
        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        # Call Unstructured.io API
        elements = await self._partition_via_api(content, filename)

        # Convert elements to chunks
        chunks = self._elements_to_chunks(elements, document_id, document_metadata)

        processing_time_ms = int((time.time() - start_time) * 1000)

        response_metadata: dict[str, Any] = {
            "processor": "unstructured-api",
            "processing_time_ms": processing_time_ms,
            "extraction_method": "unstructured-api",
            "file_size_bytes": len(content),
            "total_elements": len(elements),
        }

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    async def _partition_via_api(self, content: bytes, filename: str) -> list[dict[str, Any]]:
        """Call Unstructured.io API to partition the document.

        Args:
            content: File content as bytes
            filename: Original filename (used for format detection)

        Returns:
            List of element dictionaries from Unstructured API

        Raises:
            Exception: If API call fails
        """
        from unstructured_client import UnstructuredClient
        from unstructured_client.models import operations, shared

        client = UnstructuredClient(api_key_auth=self.config.api_key.get_secret_value())

        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=content,
                    file_name=filename,
                ),
                strategy=shared.Strategy.AUTO,
            )
        )

        log.info(
            "Calling Unstructured.io API",
            filename=filename,
            size_bytes=len(content),
        )

        resp = client.general.partition(request=req)

        if not resp.elements:
            log.warning("Unstructured.io API returned no elements", filename=filename)
            return []

        log.info(
            "Unstructured.io API returned elements",
            filename=filename,
            element_count=len(resp.elements),
        )

        # Convert response elements to list of dicts
        return [dict(elem) for elem in resp.elements]

    # element mapping to chunk with metadata, including generating chunk_id and calculating token count
    def _elements_to_chunks(
        self,
        elements: list[dict[str, Any]],
        document_id: str,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Convert Unstructured elements to OGX Chunks.

        Args:
            elements: List of element dicts from Unstructured API
            document_id: Document ID for this file
            document_metadata: Base metadata for all chunks

        Returns:
            List of OGX Chunk objects
        """
        chunks = []

        for idx, element in enumerate(elements):
            # Extract text content
            text = element.get("text", "")

            # Skip empty elements
            if not text or not text.strip():
                continue

            # Get metadata
            elem_metadata = element.get("metadata", {})
            page_number = elem_metadata.get("page_number")
            element_type = element.get("type", "Unknown")

            # Generate chunk_id from content and position
            chunk_id = generate_chunk_id(document_id, text, str(idx))

            # Calculate token count (rough estimate: split on whitespace) for num words in textok
            content_token_count = len(text.split())

            # Build metadata dict
            metadata_dict: dict[str, Any] = {
                "document_id": document_id,
                "element_type": element_type,
                "element_index": idx,
                **document_metadata,
            }
            if page_number is not None:
                metadata_dict["page_number"] = page_number

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
