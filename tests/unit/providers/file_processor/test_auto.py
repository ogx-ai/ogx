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
from ogx_api.file_processors import ProcessFileRequest


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
