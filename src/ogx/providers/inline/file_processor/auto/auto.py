# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import mimetypes
from typing import Any

from fastapi import HTTPException, UploadFile

from ogx.log import get_logger
from ogx.providers.inline.file_processor.markitdown.config import MarkItDownFileProcessorConfig
from ogx.providers.inline.file_processor.markitdown.markitdown_processor import MarkItDownFileProcessor
from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PyPDFFileProcessor
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import RetrieveFileRequest

from .config import AutoFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

# MIME types routed to MarkItDown. Derived from markitdown's bundled converters:
# DocxConverter, PptxConverter, XlsxConverter, XlsConverter, HtmlConverter,
# EpubConverter, OutlookMsgConverter, IpynbConverter, RssConverter, ImageConverter,
# AudioConverter, ZipConverter. CSV, JSON, XML, and text/* are handled by PyPDF.
MARKITDOWN_MIME_TYPES = {
    # Office documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/msword",  # .doc
    "application/vnd.ms-powerpoint",  # .ppt
    "application/vnd.ms-excel",  # .xls
    "application/rtf",  # .rtf
    # Structured formats
    "application/epub+zip",  # .epub
    "application/rss+xml",  # .rss
    # Archives
    "application/zip",  # .zip
    # Images
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
    # Audio
    "audio/mpeg",  # .mp3
    "audio/x-wav",  # .wav
}

# MIME types that docling/docling-serve handle with structure-aware parsing.
# These formats get upgraded quality (layout preservation, table detection,
# semantic chunking) when a docling provider is configured.
DOCLING_MIME_TYPES = {
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

SUPPORTED_DESCRIPTION = (
    "PDF, text (txt, csv, md, json, xml, html, code), "
    "office (DOCX, PPTX, XLSX, XLS, DOC, PPT, RTF), "
    "EPUB, RSS, ZIP, images, and audio"
)


class AutoFileProcessor:
    """Composite file processor that dispatches to backends based on MIME type.

    Always includes PyPDF (PDF/text) and MarkItDown (office/media) as built-in
    backends. Optionally dispatches to one enhanced provider when configured:

    - Docling/Docling-Serve: structure-aware parsing for PDF, DOCX, PPTX, HTML,
      and images. Other formats fall through to the built-in backends.
    - Unstructured/Unstructured-API: broad 65+ format coverage, handles all
      files when configured.

    Unsupported formats are rejected with a 422 error.
    """

    def __init__(self, config: AutoFileProcessorConfig, files_api) -> None:
        self.config = config
        self.files_api = files_api

        pypdf_config = PyPDFFileProcessorConfig(
            default_chunk_size_tokens=config.default_chunk_size_tokens,
            default_chunk_overlap_tokens=config.default_chunk_overlap_tokens,
            extract_metadata=config.extract_metadata,
            clean_text=config.clean_text,
        )
        self.pypdf = PyPDFFileProcessor(pypdf_config, files_api)

        markitdown_config = MarkItDownFileProcessorConfig(
            default_chunk_size_tokens=config.default_chunk_size_tokens,
            default_chunk_overlap_tokens=config.default_chunk_overlap_tokens,
        )
        self.markitdown = MarkItDownFileProcessor(markitdown_config, files_api)

        self.enhanced: Any = None
        self.enhanced_is_catch_all = False
        self._init_enhanced_provider(config, files_api)

    def _init_enhanced_provider(self, config: AutoFileProcessorConfig, files_api) -> None:
        if config.docling_serve is not None:
            from ogx.providers.remote.file_processor.docling_serve.docling_serve import DoclingServeFileProcessor

            self.enhanced = DoclingServeFileProcessor(config.docling_serve, files_api)
            log.info("Enhanced provider configured", provider="docling-serve")

        elif config.docling is not None:
            from ogx.providers.inline.file_processor.docling.docling import DoclingFileProcessor

            self.enhanced = DoclingFileProcessor(config.docling, files_api)
            log.info("Enhanced provider configured", provider="docling")

        elif config.unstructured_api is not None:
            from ogx.providers.remote.file_processor.unstructured_api.unstructured_api import (
                UnstructuredApiFileProcessor,
            )

            self.enhanced = UnstructuredApiFileProcessor(config.unstructured_api, files_api)
            self.enhanced_is_catch_all = True
            log.info("Enhanced provider configured", provider="unstructured-api")

        elif config.unstructured is not None:
            from ogx.providers.inline.file_processor.unstructured.unstructured import UnstructuredFileProcessor

            self.enhanced = UnstructuredFileProcessor(config.unstructured, files_api)
            self.enhanced_is_catch_all = True
            log.info("Enhanced provider configured", provider="unstructured")

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        filename = await self._resolve_filename(request, file)
        mime_type, _ = mimetypes.guess_type(filename)
        mime_category = mime_type.split("/")[0] if (mime_type and "/" in mime_type) else None

        # Catch-all enhanced providers (unstructured) handle everything
        if self.enhanced and self.enhanced_is_catch_all:
            result: ProcessFileResponse = await self.enhanced.process_file(request=request, file=file)
            return result

        # Format-specific enhanced providers (docling) handle their MIME types
        if self.enhanced and mime_type in DOCLING_MIME_TYPES:
            result = await self.enhanced.process_file(request=request, file=file)
            return result

        # Built-in backends
        if mime_type == "application/pdf" or mime_category == "text":
            return await self.pypdf.process_file(
                file=file,
                file_id=request.file_id,
                options=request.options,
                chunking_strategy=request.chunking_strategy,
            )

        if mime_type in MARKITDOWN_MIME_TYPES:
            return await self.markitdown.process_file(request=request, file=file)

        raise HTTPException(
            status_code=422,
            detail=f"File type '{mime_type or 'unknown'}' is not supported. Supported types: {SUPPORTED_DESCRIPTION}.",
        )

    async def _resolve_filename(self, request: ProcessFileRequest, file: UploadFile | None) -> str:
        if file is not None:
            name: str | None = file.filename
            if name is not None:
                return name
        if request.file_id is not None:
            file_info = await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=request.file_id))
            resolved: str = file_info.filename
            return resolved
        return "unknown"

    async def shutdown(self) -> None:
        if self.enhanced and hasattr(self.enhanced, "shutdown"):
            await self.enhanced.shutdown()
