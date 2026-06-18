# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import zipfile
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, UploadFile

from ogx.providers.inline.file_processor.markitdown.config import MarkItDownFileProcessorConfig
from ogx.providers.inline.file_processor.markitdown.markitdown_processor import MarkItDownFileProcessor
from ogx.providers.inline.file_processor.zip_utils import (
    MAX_ZIP_DECOMPRESSED_BYTES,
    MAX_ZIP_ENTRIES,
    validate_zip_content,
)


def _make_zip(num_entries: int, per_entry_bytes: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_entries):
            zf.writestr(f"file_{i}.txt", "A" * per_entry_bytes)
    return buf.getvalue()


# --- Unit tests for validate_zip_content ---


def test_validate_zip_rejects_too_many_entries():
    zip_bytes = _make_zip(MAX_ZIP_ENTRIES + 1, 1)
    with pytest.raises(HTTPException) as exc_info:
        validate_zip_content(zip_bytes, "bomb.zip")
    assert exc_info.value.status_code == 422
    assert "entries" in exc_info.value.detail.lower()


def test_validate_zip_rejects_oversized():
    per_entry = MAX_ZIP_DECOMPRESSED_BYTES // 2 + 1
    zip_bytes = _make_zip(2, per_entry)
    with pytest.raises(HTTPException) as exc_info:
        validate_zip_content(zip_bytes, "bomb.zip")
    assert exc_info.value.status_code == 422
    assert "decompressed size" in exc_info.value.detail.lower()


def test_validate_zip_accepts_small_zip():
    zip_bytes = _make_zip(2, 100)
    validate_zip_content(zip_bytes, "small.zip")


def test_validate_zip_ignores_non_zip():
    validate_zip_content(b"just plain text", "readme.txt")


# --- Integration: markitdown processor calls validation ---


@pytest.fixture
def markitdown_processor():
    config = MarkItDownFileProcessorConfig()
    files_api = MagicMock()
    return MarkItDownFileProcessor(config, files_api)


async def test_markitdown_rejects_zip_exceeding_entry_limit(markitdown_processor):
    zip_bytes = _make_zip(MAX_ZIP_ENTRIES + 1, 1)
    file = UploadFile(filename="bomb.zip", file=io.BytesIO(zip_bytes))

    with pytest.raises(HTTPException) as exc_info:
        await markitdown_processor.process_file(
            request=MagicMock(file_id=None, chunking_strategy=None),
            file=file,
        )
    assert exc_info.value.status_code == 422


async def test_markitdown_rejects_zip_exceeding_size_limit(markitdown_processor):
    per_entry = MAX_ZIP_DECOMPRESSED_BYTES // 2 + 1
    zip_bytes = _make_zip(2, per_entry)
    file = UploadFile(filename="bomb.zip", file=io.BytesIO(zip_bytes))

    with pytest.raises(HTTPException) as exc_info:
        await markitdown_processor.process_file(
            request=MagicMock(file_id=None, chunking_strategy=None),
            file=file,
        )
    assert exc_info.value.status_code == 422
