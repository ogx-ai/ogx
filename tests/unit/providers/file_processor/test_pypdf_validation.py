# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.file_processor.pypdf.pypdf import PyPDFFileProcessor


@pytest.fixture
def pypdf_processor():
    config = PyPDFFileProcessorConfig()
    files_api = MagicMock()
    return PyPDFFileProcessor(config, files_api)


async def test_rejects_docx_with_422(pypdf_processor):
    docx_bytes = b"PK\x03\x04fake_docx_content"
    file = UploadFile(filename="test.docx", file=io.BytesIO(docx_bytes))

    with pytest.raises(HTTPException) as exc_info:
        await pypdf_processor.process_file(file=file)

    assert exc_info.value.status_code == 422
    assert "not supported" in exc_info.value.detail.lower()


async def test_rejects_pptx_with_422(pypdf_processor):
    pptx_bytes = b"PK\x03\x04fake_pptx_content"
    file = UploadFile(filename="presentation.pptx", file=io.BytesIO(pptx_bytes))

    with pytest.raises(HTTPException) as exc_info:
        await pypdf_processor.process_file(file=file)

    assert exc_info.value.status_code == 422
    assert "not supported" in exc_info.value.detail.lower()


async def test_rejects_xlsx_with_422(pypdf_processor):
    xlsx_bytes = b"PK\x03\x04fake_xlsx_content"
    file = UploadFile(filename="data.xlsx", file=io.BytesIO(xlsx_bytes))

    with pytest.raises(HTTPException) as exc_info:
        await pypdf_processor.process_file(file=file)

    assert exc_info.value.status_code == 422
    assert "not supported" in exc_info.value.detail.lower()


async def test_allows_pdf(pypdf_processor):
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF"
    file = UploadFile(filename="test.pdf", file=io.BytesIO(pdf_bytes))

    result = await pypdf_processor.process_file(file=file)
    assert result is not None


async def test_allows_text_files(pypdf_processor):
    text_bytes = b"Hello, this is plain text content for testing."
    file = UploadFile(filename="readme.txt", file=io.BytesIO(text_bytes))

    result = await pypdf_processor.process_file(file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_allows_csv_files(pypdf_processor):
    csv_bytes = b"name,age\nAlice,30\nBob,25"
    file = UploadFile(filename="data.csv", file=io.BytesIO(csv_bytes))

    result = await pypdf_processor.process_file(file=file)
    assert result is not None
    assert len(result.chunks) >= 1


async def test_allows_markdown_files(pypdf_processor):
    md_bytes = b"# Hello\n\nThis is markdown."
    file = UploadFile(filename="README.md", file=io.BytesIO(md_bytes))

    result = await pypdf_processor.process_file(file=file)
    assert result is not None
    assert len(result.chunks) >= 1
