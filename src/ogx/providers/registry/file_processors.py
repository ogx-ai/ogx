# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    """Return the list of available file processor provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::auto",
            pip_packages=["chardet", "pypdf>=6.13.0", "markitdown[all]"],
            module="ogx.providers.inline.file_processor.auto",
            config_class="ogx.providers.inline.file_processor.auto.AutoFileProcessorConfig",
            api_dependencies=[Api.files],
            description=(
                "Composite file processor that dispatches to sibling providers based on file MIME "
                "type. Configure a priority list of provider IDs; each provider declares the MIME "
                "types it supports and the first match wins. Unmatched types return a 422 error. "
                "Without a priority list, falls back to built-in PyPDF (PDF, text) and MarkItDown "
                "(office, media) backends."
            ),
        ),
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["chardet", "pypdf>=6.13.0"],
            module="ogx.providers.inline.file_processor.pypdf",
            config_class="ogx.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
            api_dependencies=[Api.files],
            description="PyPDF-based file processor for extracting text content from documents.",
        ),
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::markitdown",
            pip_packages=["markitdown[all]"],
            module="ogx.providers.inline.file_processor.markitdown",
            config_class="ogx.providers.inline.file_processor.markitdown.MarkItDownFileProcessorConfig",
            api_dependencies=[Api.files],
            description="""
[MarkItDown](https://github.com/microsoft/markitdown) is a lightweight, multi-format file processor
that converts documents to Markdown using Microsoft's MarkItDown library. It supports a wide range of
document types without the heavy ML dependencies required by Docling.

## Supported Formats

- **Documents**: PDF, DOCX, PPTX, XLSX, RTF
- **Web**: HTML
- **Data**: CSV, JSON, XML
- **Code**: Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, and more
- **Text**: TXT, Markdown, RST, LaTeX

## Usage

Start OGX with the MarkItDown file processor:

```bash
ogx stack run --providers "file_processors=inline::markitdown" --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: markitdown
    provider_type: inline::markitdown
    config: {}
```

## When to Use

Choose `inline::markitdown` when you need multi-format support with minimal dependencies.
For structure-aware parsing with table, heading, and layout preservation, use `inline::docling`
or `remote::docling-serve` instead.
""",
        ),
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::docling",
            pip_packages=["docling"],
            module="ogx.providers.inline.file_processor.docling",
            config_class="ogx.providers.inline.file_processor.docling.DoclingFileProcessorConfig",
            api_dependencies=[Api.files],
            optional_api_dependencies=[Api.inference],
            description="""
[Docling](https://github.com/docling-project/docling) is a layout-aware, structure-preserving
document parser for OGX. Unlike simple text extraction, Docling understands document
structure — headings, tables, lists, and sections — and produces Markdown-formatted output that
preserves semantic boundaries. It supports PDF, DOCX, PPTX, HTML, and images.

## Features

- **Structure-aware chunking** — splits at semantic boundaries (headings, sections) using Docling's HybridChunker
- **Layout preservation** — tables, lists, and nested structures are converted to Markdown
- **Multi-format support** — PDF, DOCX, PPTX, HTML, and images
- **Better RAG quality** — structured chunks with heading metadata produce more relevant retrieval results
- **VLM-based processing** — optionally route Vision Language Model inference through the stack's model-serving
  infrastructure for richer document understanding (layout analysis, OCR via vision models)

## Usage

Start OGX with the Docling file processor using the `--providers` flag:

```bash
OLLAMA_URL=http://localhost:11434/v1 ogx stack run \\
  --providers "file_processors=inline::docling,files=inline::localfs,vector_io=inline::faiss,inference=inline::sentence-transformers,inference=remote::ollama" \\
  --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: docling
    provider_type: inline::docling
    config: {}
```

### Enabling VLM Processing

To enable VLM-based document processing, set `vlm_model` to a vision model registered with the
stack's inference API. The VLM pipeline routes inference through the stack's model-serving
infrastructure — no separate GPU resources are needed for document processing.

```yaml
file_processors:
  - provider_id: docling
    provider_type: inline::docling
    config:
      vlm_model: granite-docling-258M
      vlm_preset: granite_docling
```

When `vlm_model` is not set or no inference provider is available, the processor gracefully
degrades to the standard non-VLM pipeline.

## Installation

```bash
pip install docling
```

## Documentation

See [Docling's documentation](https://docling-project.github.io/docling/) for more details.
""",
        ),
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::unstructured",
            pip_packages=["unstructured[all-docs]>=0.21.0"],  # Security fix in 0.21.0
            module="ogx.providers.inline.file_processor.unstructured",
            config_class="ogx.providers.inline.file_processor.unstructured.UnstructuredFileProcessorConfig",
            api_dependencies=[Api.files],
            description="""
[Unstructured](https://github.com/Unstructured-IO/unstructured) is a comprehensive document
processing library supporting 65+ file formats including PDF, Office documents (DOCX, PPTX, XLSX),
email formats (EML, MSG), legacy formats (DOC, XLS), HTML, Markdown, and audio transcription.

This provider uses the local Unstructured library for offline document processing. For cloud-based
processing with better table extraction, use `remote::unstructured-api` instead.

## Features

- 65+ format support - broadest format coverage of any OGX file processor
- Email processing - EML and MSG email formats (unique to Unstructured)
- Legacy formats - DOC, XLS, and other legacy Office formats
- Audio transcription - MP3, WAV, M4A via Whisper
- Local processing - no network required, cost-effective for high volume
- Structure-aware chunking - preserves document sections and headings

## Limitations

WARNING: Table detection is unreliable in local mode (GitHub issue [#2997](https://github.com/Unstructured-IO/unstructured/issues/2997)).
For production table extraction, use `remote::unstructured-api` instead.

## System Requirements

Required system dependencies:
- `libmagic-dev` - file type detection
- `poppler-utils` - PDF processing
- `tesseract-ocr` - OCR support

Optional (for Office documents):
- `libreoffice` - Office document conversion (~800 MB)

### macOS
```bash
brew install libmagic poppler tesseract
# Optional: brew install libreoffice
```

### Ubuntu/Debian
```bash
sudo apt-get update && sudo apt-get install -y \\
    libmagic-dev \\
    poppler-utils \\
    tesseract-ocr
# Optional: sudo apt-get install -y libreoffice
```

### Docker (Recommended)
```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \\
    libmagic-dev \\
    poppler-utils \\
    tesseract-ocr \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install ogx[unstructured-local]
```

## Installation

```bash
pip install "ogx[unstructured-local]"
```

Then install system dependencies as shown above.

## Usage

Start OGX with the Unstructured file processor:

```bash
ogx stack run \\
  --providers "file_processors=inline::unstructured" \\
  --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: unstructured
    provider_type: inline::unstructured
    config:
      strategy: auto  # or 'fast', 'hi_res', 'ocr_only'
      skip_infer_table_types: ["pdf"]  # Workaround for table issues
```

## When to Use

**Use `inline::unstructured` when:**
- You need email format support (EML, MSG)
- You need legacy Office formats (DOC, XLS)
- You need audio transcription
- You need offline/local processing
- You need the broadest format coverage

**Use `inline::docling` when:**
- You need precise token-based chunking
- You need best-in-class table extraction
- You primarily process PDF/DOCX/PPTX

**Use `remote::unstructured-api` when:**
- You need reliable table extraction
- You have network connectivity and API key
- You want to avoid system dependencies

## Documentation

See [Unstructured's documentation](https://docs.unstructured.io/) for more details.
""",
        ),
        RemoteProviderSpec(
            api=Api.file_processors,
            provider_type="remote::docling-serve",
            adapter_type="docling-serve",
            pip_packages=["httpx", "docling-slim[service-client]>=2.103.0"],
            module="ogx.providers.remote.file_processor.docling_serve",
            config_class="ogx.providers.remote.file_processor.docling_serve.DoclingServeFileProcessorConfig",
            api_dependencies=[Api.files],
            description="""
[Docling Serve](https://github.com/docling-project/docling-serve) is a remote file processor that
delegates document parsing and chunking to a running Docling Serve instance. It provides the same
layout-aware, structure-preserving document conversion as the inline Docling provider, but runs as a
separate service — enabling GPU acceleration, horizontal scaling, and shared processing across
multiple OGX instances.

Docling Serve supports PDF, DOCX, PPTX, HTML, images, and more.

## Features

- **GPU-accelerated parsing** — offload document conversion to a GPU-equipped Docling Serve instance
- **Structure-aware chunking** — splits at semantic boundaries using Docling's HybridChunker
- **Layout preservation** — tables, lists, and nested structures are converted to Markdown
- **Multi-format support** — PDF, DOCX, PPTX, HTML, and images
- **Scalable architecture** — run Docling Serve as a shared service for multiple OGX instances

## Usage

Start Docling Serve (see [Docling Serve docs](https://github.com/docling-project/docling-serve/blob/main/docs/README.md) for setup):

```bash
docker run -p 5001:5001 quay.io/docling-project/docling-serve
```

Then start OGX with the remote Docling Serve provider:

```bash
DOCLING_SERVE_URL=http://localhost:5001 ogx stack run \\
  --providers "file_processors=remote::docling-serve,files=inline::localfs,vector_io=inline::faiss,inference=inline::sentence-transformers,inference=remote::ollama" \\
  --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: docling-serve
    provider_type: remote::docling-serve
    config:
      base_url: ${env.DOCLING_SERVE_URL:=http://localhost:5001}
      api_key: ${env.DOCLING_SERVE_API_KEY:=}
```

## Documentation

See [Docling Serve's documentation](https://github.com/docling-project/docling-serve/blob/main/docs/README.md) for more details on setup and configuration.
""",
        ),
        RemoteProviderSpec(
            api=Api.file_processors,
            provider_type="remote::unstructured-api",
            adapter_type="unstructured-api",
            pip_packages=[
                "unstructured-client>=0.25.0",  # >=0.25.0: supports full feature set (chunking + split_pdf_page_range)
            ],
            module="ogx.providers.remote.file_processor.unstructured_api",
            config_class="ogx.providers.remote.file_processor.unstructured_api.UnstructuredApiFileProcessorConfig",
            api_dependencies=[Api.files],
            description="""
[Unstructured.io](https://unstructured.io) is a multi-format document parser that supports 65+ file types
including emails (EML/MSG), legacy documents, presentations, spreadsheets, and more. This provider uses
the Unstructured.io SaaS API for cloud-based document processing with advanced table and image detection.

## Supported Formats

- **Documents**: PDF, DOC, DOCX, PPTX, XLSX, ODT, RTF, EPUB
- **Email**: EML, MSG (unique capability)
- **Web**: HTML, Markdown, XML, JSON
- **Images**: PNG, JPG, TIFF (with OCR)
- **Text**: TXT, CSV
- **65+ formats total** — see [Unstructured format support](https://docs.unstructured.io/pipelines/supported-file-types)

## Features

- **Multi-format support** — 65+ file types including email formats (EML/MSG)
- **Cloud-based processing** — no local dependencies or system requirements
- **Table detection** — extracts tables with structure preservation
- **Image detection** — identifies and extracts image elements
- **SOC2/HIPAA/GDPR certified** — suitable for regulated industries

## Usage

Get an API key from [Unstructured.io](https://unstructured.io) (free tier available), then start OGX:

```bash
UNSTRUCTURED_API_KEY=your-api-key ogx stack run \\
  --providers "file_processors=remote::unstructured-api,files=inline::localfs,vector_io=inline::faiss,inference=inline::sentence-transformers,inference=remote::ollama" \\
  --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: unstructured
    provider_type: remote::unstructured-api
    config:
      api_key: ${env.UNSTRUCTURED_API_KEY}
```

## When to Use

- **Diverse formats**: Need to process emails, legacy documents, or 10+ different file types
- **Managed service**: Want zero setup and no system dependencies
- **Compliance**: Require SOC2/HIPAA/GDPR certified processing
- **Email RAG**: Building customer support or communication archive applications

For faster processing with fewer formats, use `inline::docling` instead.

## Performance

- Processing speed: ~1-2 seconds per page
- Best for: Documents under 100 pages
- Cost: ~$0.01 per page (verify current pricing with Unstructured.io)

## Documentation

See [Unstructured.io documentation](https://docs.unstructured.io) for API details and format support.
""",
        ),
    ]
