# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import mimetypes

from fastapi import HTTPException, UploadFile

from ogx.providers.inline.file_processor.markitdown.config import MarkItDownFileProcessorConfig
from ogx.providers.inline.file_processor.markitdown.markitdown_processor import MarkItDownFileProcessor
from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PyPDFFileProcessor
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.files import RetrieveFileRequest

from .config import AutoFileProcessorConfig

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

SUPPORTED_DESCRIPTION = (
    "PDF, text (txt, csv, md, json, xml, html, code), "
    "office (DOCX, PPTX, XLSX, XLS, DOC, PPT, RTF), "
    "EPUB, RSS, ZIP, images, and audio"
)


class AutoFileProcessor:
    """Composite file processor that dispatches to backends based on MIME type.

    Routes PDF and text files to PyPDF. Office documents, images, audio, and
    other rich formats are routed to MarkItDown. Unsupported formats are
    rejected with a 422 error listing the supported types.
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

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        filename = await self._resolve_filename(request, file)
        mime_type, _ = mimetypes.guess_type(filename)
        mime_category = mime_type.split("/")[0] if (mime_type and "/" in mime_type) else None

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
        pass
