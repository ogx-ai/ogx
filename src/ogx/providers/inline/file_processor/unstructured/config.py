# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class UnstructuredFileProcessorConfig(BaseModel):
    """Configuration for local Unstructured file processor.

    Supports 65+ file formats including PDF, DOCX, PPTX, XLSX, EML, MSG, HTML,
    Markdown, audio transcription, and more via the local Unstructured library.

    System dependencies required:
    - libmagic (file type detection)
    - poppler-utils (PDF processing)
    - tesseract-ocr (OCR support)
    - libreoffice (optional, for Office document conversion)

    Docker installation recommended for production deployments.
    """

    strategy: Literal["auto", "fast", "hi_res", "ocr_only"] = Field(
        default="auto",
        description=(
            "Partitioning strategy for document processing. "
            "'auto' (default) intelligently selects the best approach based on document type. "
            "'fast' uses text extraction without layout analysis (fastest). "
            "'hi_res' uses layout models for better structure detection (slowest). "
            "'ocr_only' uses Tesseract OCR for scanned documents. "
            "WARNING: Table detection is unreliable in local mode due to known issue "
            "(https://github.com/Unstructured-IO/unstructured/issues/2997). "
            "Use remote::unstructured-api for production table extraction."
        ),
    )

    default_chunk_size_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["max_chunk_size_tokens"].default,
        ge=100,
        le=4096,
        description="Default chunk size in tokens when chunking_strategy type is 'auto'",
    )

    default_chunk_overlap_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["chunk_overlap_tokens"].default,
        ge=0,
        le=2048,
        description="Default chunk overlap in tokens when chunking_strategy type is 'auto'",
    )

    include_page_breaks: bool = Field(
        default=True,
        description="Include PageBreak elements in output for supported formats (PDF, PPTX, HTML)",
    )

    skip_infer_table_types: list[str] = Field(
        default_factory=lambda: ["pdf"],
        description=(
            "File types to skip table inference for (workaround for local table detection issues). "
            "Example: ['pdf', 'docx']. Set to empty list [] to attempt table detection for all formats. "
            "Note: Table detection is unreliable in local mode; use remote::unstructured-api for reliable tables."
        ),
    )

    extract_images_in_pdf: bool = Field(
        default=False,
        description=(
            "Extract images from PDFs. Requires strategy='hi_res'. "
            "May fail on some systems due to missing dependencies. "
            "Set to True only if you need image extraction and have verified it works in your environment."
        ),
    )

    languages: list[str] = Field(
        default_factory=lambda: ["eng"],
        description=(
            "OCR language codes for Tesseract (e.g., ['eng', 'spa', 'deu']). "
            "Additional language packs must be installed separately via tesseract-ocr-{lang}."
        ),
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        """Sample configuration for running the provider."""
        return {
            "strategy": "auto",
            "default_chunk_size_tokens": 800,
            "default_chunk_overlap_tokens": 400,
            "skip_infer_table_types": ["pdf"],
            "languages": ["eng"],
        }
