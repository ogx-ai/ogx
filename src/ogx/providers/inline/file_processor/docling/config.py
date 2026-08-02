# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class DoclingFileProcessorConfig(BaseModel):
    """Configuration for Docling file processor."""

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

    do_ocr: bool = Field(
        default=True,
        description=(
            "Enable OCR for scanned documents. Set to False for digital PDFs "
            "(with embedded text) to improve processing speed by ~3x for non-scanned PDFs. "
            "Note: Setting to False on scanned PDFs will result in minimal text extraction. "
            "Ignored when vlm_model is set (VLM pipeline handles text extraction)."
        ),
    )

    vlm_model: str | None = Field(
        default=None,
        description=(
            "Model identifier for VLM-based document processing. When set and an inference "
            "provider is available, enables the VLM pipeline for richer document understanding "
            "(layout analysis, OCR via vision models). The model must be registered with the "
            "stack's inference API. When None (default), uses the standard non-VLM pipeline."
        ),
    )

    vlm_preset: str = Field(
        default="granite_docling",
        description=(
            "Docling VLM preset controlling prompt template and response format. "
            "Must be a valid preset name registered with VlmConvertOptions. "
            "The default 'granite_docling' is the recommended preset for production use."
        ),
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "default_chunk_size_tokens": 800,
            "default_chunk_overlap_tokens": 400,
            "do_ocr": True,
            "vlm_model": None,
            "vlm_preset": "granite_docling",
        }
