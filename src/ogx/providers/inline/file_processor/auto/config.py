# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig
from ogx.providers.inline.file_processor.unstructured.config import UnstructuredFileProcessorConfig
from ogx.providers.remote.file_processor.docling_serve.config import DoclingServeFileProcessorConfig
from ogx.providers.remote.file_processor.unstructured_api.config import UnstructuredApiFileProcessorConfig
from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig

ENHANCED_PROVIDER_FIELDS = ("docling", "docling_serve", "unstructured", "unstructured_api")


class AutoFileProcessorConfig(BaseModel):
    """Configuration for the auto file processor.

    The auto file processor dispatches to the appropriate backend based on file
    MIME type. It always includes PyPDF for PDF and text files and MarkItDown
    for office/media formats.

    Optionally, one enhanced provider can be configured for higher-quality or
    broader-coverage processing:

    - ``docling`` / ``docling_serve``: Structure-aware parsing for PDF, DOCX,
      PPTX, HTML, and images. Formats outside this set fall through to the
      built-in PyPDF/MarkItDown backends.
    - ``unstructured`` / ``unstructured_api``: Broad 65+ format coverage.
      Replaces the built-in backends entirely when configured.

    At most one enhanced provider may be set at a time.
    """

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

    extract_metadata: bool = Field(default=True, description="Whether to extract PDF metadata (title, author, etc.)")

    clean_text: bool = Field(
        default=True, description="Whether to clean extracted text (remove extra whitespace, normalize line breaks)"
    )

    docling: DoclingFileProcessorConfig | None = Field(
        default=None,
        description=(
            "Enable inline Docling for structure-aware PDF, DOCX, PPTX, HTML, "
            "and image processing. Requires the 'docling' package."
        ),
    )
    docling_serve: DoclingServeFileProcessorConfig | None = Field(
        default=None,
        description=(
            "Enable remote Docling Serve for GPU-accelerated, structure-aware "
            "document processing. Requires a running Docling Serve instance."
        ),
    )
    unstructured: UnstructuredFileProcessorConfig | None = Field(
        default=None,
        description=(
            "Enable inline Unstructured for 65+ format support including email "
            "(EML, MSG) and legacy Office formats. Requires the 'unstructured' "
            "package and system dependencies."
        ),
    )
    unstructured_api: UnstructuredApiFileProcessorConfig | None = Field(
        default=None,
        description=(
            "Enable Unstructured.io SaaS API for 65+ format support with "
            "cloud-based processing. Requires an API key from unstructured.io."
        ),
    )

    @model_validator(mode="after")
    def validate_single_enhanced_provider(self) -> Self:
        configured = [name for name in ENHANCED_PROVIDER_FIELDS if getattr(self, name) is not None]
        if len(configured) > 1:
            raise ValueError(f"At most one enhanced provider can be configured, but found: {', '.join(configured)}")
        return self

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {}
