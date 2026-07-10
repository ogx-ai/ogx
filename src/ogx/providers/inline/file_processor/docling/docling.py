# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import tempfile
import threading
import time
import uuid
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from fastapi import UploadFile

from ogx.log import get_logger
from ogx.providers.inline.file_processor.docling.vlm_engine import OgxInferenceVlmEngine
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

from .config import DoclingFileProcessorConfig

log = get_logger(name=__name__, category="providers::file_processors")

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


class DoclingFileProcessor:
    """Docling-based file processor with structure-aware chunking.

    Supports multiple file formats via docling's DocumentConverter (PDF, DOCX, PPTX, HTML, images, etc.).
    """

    def __init__(self, config: DoclingFileProcessorConfig, files_api=None, inference_api=None) -> None:
        self.config = config
        self.files_api = files_api
        self.inference_api = inference_api
        self._vlm_enabled = False

        self.converter = self._build_converter()
        self._converter_lock = threading.Lock()

    def supported_mime_types(self) -> set[str] | None:
        return DOCLING_MIME_TYPES

    def _build_converter(self) -> DocumentConverter:
        if self.config.vlm_model and self.inference_api:
            return self._build_vlm_converter()

        if self.config.vlm_model and not self.inference_api:
            log.warning(
                "vlm_model is configured but no inference provider is available, falling back to standard pipeline",
                vlm_model=self.config.vlm_model,
            )

        pipeline_options = PdfPipelineOptions(do_ocr=self.config.do_ocr)
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    def _build_vlm_converter(self) -> DocumentConverter:
        assert self.config.vlm_model is not None

        available_presets = list(VlmConvertOptions._presets.keys())
        if self.config.vlm_preset not in available_presets:
            raise ValueError(f"Invalid vlm_preset '{self.config.vlm_preset}'. Available presets: {available_presets}")

        vlm_options = VlmConvertOptions.from_preset(
            self.config.vlm_preset,
            engine_options=ApiVlmEngineOptions(engine_type=VlmEngineType.API_OPENAI),
        )

        vlm_pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_pipeline_options,
                )
            }
        )

        converter.initialize_pipeline(InputFormat.PDF)

        event_loop = asyncio.get_event_loop()
        ogx_engine = OgxInferenceVlmEngine(
            inference_api=self.inference_api,
            model=self.config.vlm_model,
            event_loop=event_loop,
        )

        for pipeline in converter.initialized_pipelines.values():
            for stage in getattr(pipeline, "build_pipe", []):
                if hasattr(stage, "engine"):
                    stage.engine = ogx_engine
                    break

        self._vlm_enabled = True
        log.info(
            "VLM pipeline enabled",
            vlm_model=self.config.vlm_model,
            vlm_preset=self.config.vlm_preset,
        )

        return converter

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using docling and return chunks."""
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

        return await asyncio.to_thread(self._process_content, content, filename, file_id, chunking_strategy, start_time)

    def _process_content(
        self,
        content: bytes,
        filename: str,
        file_id: str | None,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        start_time: float,
    ) -> ProcessFileResponse:
        """Convert and chunk file content. Runs in a thread."""
        validate_zip_content(content, filename)

        # Preserve original file extension so DocumentConverter can detect the format
        suffix = os.path.splitext(filename)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()

            with self._converter_lock:
                result = self.converter.convert(tmp.name)

        doc = result.document
        page_count = doc.num_pages()

        document_id = str(uuid.uuid4())

        document_metadata: dict[str, Any] = {"filename": filename}
        if file_id:
            document_metadata["file_id"] = file_id

        processing_time_ms = int((time.time() - start_time) * 1000)

        extraction_method = "docling-vlm" if self._vlm_enabled else "docling"
        response_metadata: dict[str, Any] = {
            "processor": "docling",
            "processing_time_ms": processing_time_ms,
            "page_count": page_count,
            "extraction_method": extraction_method,
            "file_size_bytes": len(content),
        }
        if self._vlm_enabled:
            response_metadata["vlm_model"] = self.config.vlm_model
            response_metadata["vlm_preset"] = self.config.vlm_preset

        # Create chunks
        chunks = self._create_chunks(doc, document_id, chunking_strategy, document_metadata)

        return ProcessFileResponse(chunks=chunks, metadata=response_metadata)

    def _create_chunks(
        self,
        doc: Any,
        document_id: str,
        chunking_strategy: VectorStoreChunkingStrategy | None,
        document_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Create chunks from a docling Document.

        Chunking semantics:
        - chunking_strategy is None -> return all text as a single chunk
        - chunking_strategy.type == "auto" -> HybridChunker with configured defaults
        - chunking_strategy.type == "static" -> HybridChunker with provided max_tokens
        """
        if not chunking_strategy:
            # No chunking - collect all text as a single chunk
            text = doc.export_to_markdown()
            if not text or not text.strip():
                return []

            chunk_id = generate_chunk_id(document_id, text)
            return [
                Chunk(
                    content=text,
                    chunk_id=chunk_id,
                    metadata={
                        "document_id": document_id,
                        **document_metadata,
                    },
                    chunk_metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source=document_metadata.get("filename", ""),
                        content_token_count=len(text.split()),
                    ),
                )
            ]

        # Determine max_tokens based on strategy
        if chunking_strategy.type == "auto":
            max_tokens = self.config.default_chunk_size_tokens
        elif chunking_strategy.type == "static":
            max_tokens = chunking_strategy.static.max_chunk_size_tokens
        else:
            max_tokens = self.config.default_chunk_size_tokens

        # max_tokens is set on the tokenizer, not on HybridChunker directly
        default_chunker = HybridChunker()
        tokenizer = HuggingFaceTokenizer(
            tokenizer=default_chunker.tokenizer.tokenizer,  # type: ignore[attr-defined]
            max_tokens=max_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer)
        doc_chunks = list(chunker.chunk(doc))

        if not doc_chunks:
            return []

        chunks: list[Chunk] = []
        for i, doc_chunk in enumerate(doc_chunks):
            text = doc_chunk.text
            if not text or not text.strip():
                continue

            headings = getattr(doc_chunk, "headings", None)
            chunk_window = f"{i}"

            chunk_id = generate_chunk_id(document_id, text, chunk_window)

            meta: dict[str, Any] = {
                "document_id": document_id,
                **document_metadata,
            }
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
