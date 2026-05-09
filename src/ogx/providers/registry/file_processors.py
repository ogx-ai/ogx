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
            pip_packages=["pypdf>=6.7.2", "markitdown[all]"],
            module="ogx.providers.inline.file_processor.auto",
            config_class="ogx.providers.inline.file_processor.auto.AutoFileProcessorConfig",
            api_dependencies=[Api.files],
            description=(
                "Composite file processor that automatically dispatches to the appropriate backend "
                "based on file MIME type. Routes PDF and text files to PyPDF, and office/structured "
                "formats (DOCX, PPTX, XLSX, HTML, JSON, XML) to MarkItDown when installed. "
                "Unsupported formats are rejected with a clear error listing the supported types."
            ),
        ),
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["pypdf>=6.7.2"],
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

## Installation

```bash
pip install docling
```

## Documentation

See [Docling's documentation](https://docling-project.github.io/docling/) for more details.
""",
        ),
        RemoteProviderSpec(
            api=Api.file_processors,
            provider_type="remote::docling-serve",
            adapter_type="docling-serve",
            pip_packages=["httpx"],
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
DOCLING_SERVE_URL=http://localhost:5001/v1 ogx stack run \\
  --providers "file_processors=remote::docling-serve,files=inline::localfs,vector_io=inline::faiss,inference=inline::sentence-transformers,inference=remote::ollama" \\
  --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: docling-serve
    provider_type: remote::docling-serve
    config:
      base_url: ${env.DOCLING_SERVE_URL:=http://localhost:5001/v1}
      api_key: ${env.DOCLING_SERVE_API_KEY:=}
```

## Documentation

See [Docling Serve's documentation](https://github.com/docling-project/docling-serve/blob/main/docs/README.md) for more details on setup and configuration.
""",
        ),
    ]
