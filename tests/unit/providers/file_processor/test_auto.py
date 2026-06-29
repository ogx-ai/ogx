# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from ogx.providers.inline.file_processor.auto.auto import AutoFileProcessor
from ogx.providers.inline.file_processor.auto.config import AutoFileProcessorConfig
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse


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


# --- Default dispatch tests (no enhanced provider) ---


async def test_routes_pdf_to_pypdf(auto_processor):
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF"
    file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None


async def test_routes_text_to_pypdf(auto_processor):
    text_bytes = b"Hello, this is plain text."
    file = UploadFile(filename="readme.txt", file=io.BytesIO(text_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_csv_to_pypdf(auto_processor):
    csv_bytes = b"name,age\nAlice,30\nBob,25"
    file = UploadFile(filename="data.csv", file=io.BytesIO(csv_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_markdown_to_pypdf(auto_processor):
    md_bytes = b"# Hello\n\nThis is markdown."
    file = UploadFile(filename="README.md", file=io.BytesIO(md_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_routes_docx_to_markitdown(auto_processor):
    docx_bytes = b"PK\x03\x04fake_docx_content"
    file = UploadFile(filename="test.docx", file=io.BytesIO(docx_bytes))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)

    assert exc_info.value.status_code == 422
    assert "Failed to process file" in exc_info.value.detail


async def test_routes_pptx_to_markitdown(auto_processor):
    pptx_bytes = b"PK\x03\x04fake_pptx_content"
    file = UploadFile(filename="presentation.pptx", file=io.BytesIO(pptx_bytes))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)

    assert exc_info.value.status_code == 422
    assert "Failed to process file" in exc_info.value.detail


async def test_routes_xlsx_to_markitdown(auto_processor):
    xlsx_bytes = b"PK\x03\x04fake_xlsx_content"
    file = UploadFile(filename="data.xlsx", file=io.BytesIO(xlsx_bytes))
    request = ProcessFileRequest()

    result = await auto_processor.process_file(request, file=file)
    assert result is not None
    assert result.metadata["processor"] == "markitdown"


async def test_rejects_unsupported_format_with_422(auto_processor):
    file = UploadFile(filename="test.xyz", file=io.BytesIO(b"some data"))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await auto_processor.process_file(request, file=file)

    assert exc_info.value.status_code == 422
    detail = exc_info.value.detail.lower()
    assert "not supported" in detail
    assert "pdf" in detail


async def test_routes_file_id_using_resolved_filename(auto_processor_with_files_api):
    request = ProcessFileRequest(file_id="file-123456")

    result = await auto_processor_with_files_api.process_file(request)
    assert result is not None
    assert len(result.chunks) >= 1


# --- Enhanced provider: docling ---


async def test_docling_routes_pdf():
    mock_response = ProcessFileResponse(chunks=[], metadata={"processor": "docling"})
    mock_docling = MagicMock()
    mock_docling.process_file = AsyncMock(return_value=mock_response)

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_docling
    processor.enhanced_is_catch_all = False

    file = UploadFile(filename="report.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "docling"
    mock_docling.process_file.assert_called_once()


async def test_docling_routes_docx():
    mock_response = ProcessFileResponse(chunks=[], metadata={"processor": "docling"})
    mock_docling = MagicMock()
    mock_docling.process_file = AsyncMock(return_value=mock_response)

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_docling
    processor.enhanced_is_catch_all = False

    file = UploadFile(filename="document.docx", file=io.BytesIO(b"docx data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "docling"


async def test_docling_routes_html():
    mock_response = ProcessFileResponse(chunks=[], metadata={"processor": "docling"})
    mock_docling = MagicMock()
    mock_docling.process_file = AsyncMock(return_value=mock_response)

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_docling
    processor.enhanced_is_catch_all = False

    file = UploadFile(filename="page.html", file=io.BytesIO(b"<html>content</html>"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "docling"


async def test_docling_falls_through_for_xlsx():
    """XLSX is not in DOCLING_MIME_TYPES, so it falls through to markitdown."""
    mock_docling = MagicMock()
    mock_docling.process_file = AsyncMock()

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_docling
    processor.enhanced_is_catch_all = False

    xlsx_bytes = b"PK\x03\x04fake_xlsx_content"
    file = UploadFile(filename="data.xlsx", file=io.BytesIO(xlsx_bytes))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    mock_docling.process_file.assert_not_called()
    assert result.metadata["processor"] == "markitdown"


async def test_docling_falls_through_for_text():
    """Plain text falls through to pypdf even when docling is configured."""
    mock_docling = MagicMock()
    mock_docling.process_file = AsyncMock()

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_docling
    processor.enhanced_is_catch_all = False

    file = UploadFile(filename="readme.txt", file=io.BytesIO(b"Hello"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    mock_docling.process_file.assert_not_called()
    assert len(result.chunks) >= 1


async def test_docling_rejects_unsupported_format():
    """Formats not handled by docling or built-ins get 422."""
    mock_docling = MagicMock()
    mock_docling.process_file = AsyncMock()

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_docling
    processor.enhanced_is_catch_all = False

    file = UploadFile(filename="email.eml", file=io.BytesIO(b"email data"))
    request = ProcessFileRequest()

    with pytest.raises(HTTPException) as exc_info:
        await processor.process_file(request, file=file)
    assert exc_info.value.status_code == 422


# --- Enhanced provider: unstructured (catch-all) ---


async def test_unstructured_handles_pdf():
    mock_response = ProcessFileResponse(chunks=[], metadata={"processor": "unstructured"})
    mock_unstructured = MagicMock()
    mock_unstructured.process_file = AsyncMock(return_value=mock_response)

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_unstructured
    processor.enhanced_is_catch_all = True

    file = UploadFile(filename="report.pdf", file=io.BytesIO(b"pdf data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "unstructured"
    mock_unstructured.process_file.assert_called_once()


async def test_unstructured_handles_eml():
    mock_response = ProcessFileResponse(chunks=[], metadata={"processor": "unstructured"})
    mock_unstructured = MagicMock()
    mock_unstructured.process_file = AsyncMock(return_value=mock_response)

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_unstructured
    processor.enhanced_is_catch_all = True

    file = UploadFile(filename="message.eml", file=io.BytesIO(b"email data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "unstructured"


async def test_unstructured_handles_unknown_format():
    """Catch-all providers handle any format without 422."""
    mock_response = ProcessFileResponse(chunks=[], metadata={"processor": "unstructured"})
    mock_unstructured = MagicMock()
    mock_unstructured.process_file = AsyncMock(return_value=mock_response)

    config = AutoFileProcessorConfig()
    files_api = MagicMock()
    processor = AutoFileProcessor(config, files_api)
    processor.enhanced = mock_unstructured
    processor.enhanced_is_catch_all = True

    file = UploadFile(filename="archive.7z", file=io.BytesIO(b"archive data"))
    request = ProcessFileRequest()
    result = await processor.process_file(request, file=file)

    assert result.metadata["processor"] == "unstructured"


# --- Enhanced provider initialization ---


async def test_init_with_docling_config():
    """Verify docling provider is initialized when config is set."""
    mock_module = MagicMock()
    mock_processor_cls = MagicMock()
    mock_module.DoclingFileProcessor = mock_processor_cls

    from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig

    config = AutoFileProcessorConfig(docling=DoclingFileProcessorConfig())
    files_api = MagicMock()

    with patch.dict("sys.modules", {"ogx.providers.inline.file_processor.docling.docling": mock_module}):
        processor = AutoFileProcessor(config, files_api)

    assert processor.enhanced is mock_processor_cls.return_value
    assert processor.enhanced_is_catch_all is False


async def test_init_with_docling_serve_config():
    """Verify docling-serve provider is initialized when config is set."""
    mock_module = MagicMock()
    mock_processor_cls = MagicMock()
    mock_module.DoclingServeFileProcessor = mock_processor_cls

    from ogx.providers.remote.file_processor.docling_serve.config import DoclingServeFileProcessorConfig

    config = AutoFileProcessorConfig(docling_serve=DoclingServeFileProcessorConfig())
    files_api = MagicMock()

    with patch.dict("sys.modules", {"ogx.providers.remote.file_processor.docling_serve.docling_serve": mock_module}):
        processor = AutoFileProcessor(config, files_api)

    assert processor.enhanced is mock_processor_cls.return_value
    assert processor.enhanced_is_catch_all is False


async def test_init_with_unstructured_api_config():
    """Verify unstructured-api provider is initialized as catch-all."""
    mock_module = MagicMock()
    mock_processor_cls = MagicMock()
    mock_module.UnstructuredApiFileProcessor = mock_processor_cls

    from ogx.providers.remote.file_processor.unstructured_api.config import UnstructuredApiFileProcessorConfig

    config = AutoFileProcessorConfig(unstructured_api=UnstructuredApiFileProcessorConfig(api_key="test-key"))
    files_api = MagicMock()

    with patch.dict(
        "sys.modules", {"ogx.providers.remote.file_processor.unstructured_api.unstructured_api": mock_module}
    ):
        processor = AutoFileProcessor(config, files_api)

    assert processor.enhanced is mock_processor_cls.return_value
    assert processor.enhanced_is_catch_all is True


# --- Config validation ---


def test_config_rejects_multiple_enhanced_providers():
    from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig
    from ogx.providers.inline.file_processor.unstructured.config import UnstructuredFileProcessorConfig

    with pytest.raises(ValueError, match="At most one enhanced provider"):
        AutoFileProcessorConfig(
            docling=DoclingFileProcessorConfig(),
            unstructured=UnstructuredFileProcessorConfig(),
        )


def test_config_allows_single_enhanced_provider():
    from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig

    config = AutoFileProcessorConfig(docling=DoclingFileProcessorConfig())
    assert config.docling is not None
    assert config.docling_serve is None
    assert config.unstructured is None
    assert config.unstructured_api is None


def test_config_allows_no_enhanced_provider():
    config = AutoFileProcessorConfig()
    assert config.docling is None
    assert config.docling_serve is None
    assert config.unstructured is None
    assert config.unstructured_api is None
