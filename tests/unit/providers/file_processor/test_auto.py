# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from ogx.providers.inline.file_processor.auto.auto import AutoFileProcessor
from ogx.providers.inline.file_processor.auto.config import AutoFileProcessorConfig
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse

# --- Helpers ---


def _make_provider(name: str, mime_types: set[str] | None = None):
    """Create a mock provider with supported_mime_types and process_file."""
    response = ProcessFileResponse(chunks=[], metadata={"processor": name})
    provider = MagicMock()
    provider.process_file = AsyncMock(return_value=response)
    provider.supported_mime_types = MagicMock(return_value=mime_types)
    provider.__provider_id__ = name
    return provider


# --- Legacy mode tests (no priority, no siblings) ---


@pytest.fixture
def auto_processor():
    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    return AutoFileProcessor(config, files_api)


@pytest.fixture
def auto_processor_with_files_api():
    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    file_info = MagicMock()
    file_info.filename = "document.txt"
    files_api.openai_retrieve_file = AsyncMock(return_value=file_info)

    content_response = MagicMock()
    content_response.body = b"Hello from file storage."
    files_api.openai_retrieve_file_content = AsyncMock(return_value=content_response)

    return AutoFileProcessor(config, files_api)


async def test_legacy_routes_pdf_to_pypdf(auto_processor):
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF"
    file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_bytes))
    request = ProcessFileRequest()
    result = await auto_processor.process_file(request, file=file)
    assert result is not None


async def test_legacy_routes_text_to_pypdf(auto_processor):
    file = UploadFile(filename="readme.txt", file=io.BytesIO(b"Hello, this is plain text."))
    request = ProcessFileRequest()
    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_legacy_routes_csv_to_pypdf(auto_processor):
    file = UploadFile(filename="data.csv", file=io.BytesIO(b"name,age\nAlice,30"))
    request = ProcessFileRequest()
    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_legacy_routes_docx_to_markitdown(auto_processor):
    file = UploadFile(filename="test.docx", file=io.BytesIO(b"PK\x03\x04fake_docx_content"))
    request = ProcessFileRequest()
    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)
    assert exc_info.value.status_code == 422
    assert "Failed to process file" in exc_info.value.detail


async def test_legacy_routes_xlsx_to_markitdown(auto_processor):
    file = UploadFile(filename="data.xlsx", file=io.BytesIO(b"PK\x03\x04fake_xlsx_content"))
    request = ProcessFileRequest()
    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert result.metadata["processor"] == "markitdown"


async def test_legacy_rejects_unsupported_format(auto_processor):
    file = UploadFile(filename="test.xyz", file=io.BytesIO(b"some data"))
    request = ProcessFileRequest()
    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)
    assert exc_info.value.status_code == 422
    assert "not supported" in exc_info.value.detail.lower()


async def test_legacy_routes_file_id(auto_processor_with_files_api):
    request = ProcessFileRequest(file_id="file-123456")
    result = await auto_processor_with_files_api.process_file(request)
    assert result is not None
    assert len(result.chunks) >= 1


# --- Priority dispatch tests ---


async def test_priority_exact_mime_match():
    docling = _make_provider("docling", {"application/pdf", "text/html"})
    pypdf = _make_provider("pypdf", {"application/pdf", "text/*"})

    config = AutoFileProcessorConfig(priority=["docling", "pypdf"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"docling": docling, "pypdf": pypdf})

    file = UploadFile(filename="report.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "docling"
    docling.process_file.assert_called_once()
    pypdf.process_file.assert_not_called()


async def test_priority_first_provider_wins():
    """When two providers support the same MIME type, the first in priority wins."""
    provider_a = _make_provider("provider_a", {"application/pdf"})
    provider_b = _make_provider("provider_b", {"application/pdf"})

    config = AutoFileProcessorConfig(priority=["provider_a", "provider_b"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"provider_a": provider_a, "provider_b": provider_b})

    file = UploadFile(filename="test.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "provider_a"


async def test_priority_fallthrough_to_second_provider():
    """When first provider doesn't support the type, second provider handles it."""
    docling = _make_provider("docling", {"application/pdf"})
    markitdown = _make_provider(
        "markitdown",
        {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        },
    )

    config = AutoFileProcessorConfig(priority=["docling", "markitdown"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"docling": docling, "markitdown": markitdown})

    file = UploadFile(filename="data.xlsx", file=io.BytesIO(b"xlsx data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "markitdown"
    docling.process_file.assert_not_called()


async def test_priority_wildcard_category_match():
    """Providers declaring text/* match any text subtype."""
    pypdf = _make_provider("pypdf", {"application/pdf", "text/*"})

    config = AutoFileProcessorConfig(priority=["pypdf"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"pypdf": pypdf})

    file = UploadFile(filename="data.csv", file=io.BytesIO(b"a,b\n1,2"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "pypdf"


async def test_priority_exact_match_beats_wildcard():
    """Exact MIME match takes priority over wildcard category match."""
    html_provider = _make_provider("html_provider", {"text/html"})
    text_provider = _make_provider("text_provider", {"text/*"})

    config = AutoFileProcessorConfig(priority=["html_provider", "text_provider"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"html_provider": html_provider, "text_provider": text_provider})

    file = UploadFile(filename="page.html", file=io.BytesIO(b"<html></html>"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "html_provider"


async def test_priority_skips_provider_returning_none():
    """Providers returning None from supported_mime_types() are skipped."""
    good = _make_provider("good", {"application/pdf"})
    bad = _make_provider("bad", None)

    config = AutoFileProcessorConfig(priority=["bad", "good"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"bad": bad, "good": good})

    file = UploadFile(filename="report.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "good"
    bad.process_file.assert_not_called()


async def test_priority_skips_provider_missing_method():
    """Providers without supported_mime_types() are skipped."""
    good = _make_provider("good", {"application/pdf"})
    bad = MagicMock(spec=[])

    config = AutoFileProcessorConfig(priority=["bad", "good"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"bad": bad, "good": good})

    file = UploadFile(filename="report.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "good"


async def test_priority_no_match_raises_422():
    """When no provider matches and no catch-all exists, raises 422."""
    docling = _make_provider("docling", {"application/pdf"})

    config = AutoFileProcessorConfig(priority=["docling"])
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"docling": docling})

    file = UploadFile(filename="message.eml", file=io.BytesIO(b"email data"))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await processor.process_file(request, file=file)
    assert exc_info.value.status_code == 422


async def test_priority_missing_provider_raises_error():
    """Referencing a non-existent provider ID raises ValueError."""
    pypdf = _make_provider("pypdf", {"application/pdf"})

    config = AutoFileProcessorConfig(priority=["pypdf", "nonexistent"])
    processor = AutoFileProcessor(config, MagicMock())
    with pytest.raises(ValueError, match="Failed to resolve priority entry 'nonexistent'"):
        processor.set_sibling_providers({"pypdf": pypdf})


# --- Auto-discovery tests (no priority, siblings injected) ---


async def test_autodiscover_uses_siblings_in_order():
    """Without priority, auto discovers siblings and uses them in config order."""
    docling = _make_provider("docling", {"application/pdf", "text/html"})
    pypdf = _make_provider("pypdf", {"application/pdf", "text/*"})

    config = AutoFileProcessorConfig()
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"docling": docling, "pypdf": pypdf})

    file = UploadFile(filename="report.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "docling"


async def test_autodiscover_fallthrough():
    """Auto-discovered siblings fall through to the next provider."""
    docling = _make_provider("docling", {"application/pdf"})
    markitdown = _make_provider("markitdown", {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})

    config = AutoFileProcessorConfig()
    processor = AutoFileProcessor(config, MagicMock())
    processor.set_sibling_providers({"docling": docling, "markitdown": markitdown})

    file = UploadFile(filename="data.xlsx", file=io.BytesIO(b"xlsx data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "markitdown"


async def test_no_siblings_stays_in_legacy_mode():
    """Without siblings, auto stays in legacy mode."""
    config = AutoFileProcessorConfig(priority=["docling"])
    processor = AutoFileProcessor(config, MagicMock())

    file = UploadFile(filename="readme.txt", file=io.BytesIO(b"Hello, this is plain text."))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


# --- supported_mime_types on individual providers ---


def test_pypdf_supported_mime_types():
    from ogx.providers.inline.file_processor.pypdf.adapter import PyPDFFileProcessorAdapter
    from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig

    adapter = PyPDFFileProcessorAdapter(PyPDFFileProcessorConfig(), MagicMock())
    types = adapter.supported_mime_types()
    assert types is not None
    assert "application/pdf" in types
    assert "text/*" in types


def test_markitdown_supported_mime_types():
    from ogx.providers.inline.file_processor.markitdown.config import MarkItDownFileProcessorConfig
    from ogx.providers.inline.file_processor.markitdown.markitdown_processor import (
        MARKITDOWN_MIME_TYPES,
        MarkItDownFileProcessor,
    )

    processor = MarkItDownFileProcessor(MarkItDownFileProcessorConfig(), MagicMock())
    types = processor.supported_mime_types()
    assert types is MARKITDOWN_MIME_TYPES
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in types
