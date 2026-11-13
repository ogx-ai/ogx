# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io

import pytest
from fastapi import UploadFile

from ogx_api.common.errors import FileTooLargeError
from ogx_api.common.upload_limits import (
    PreReadUploadFile,
    enforce_upload_size_limit,
    read_upload_with_size_limit,
)


def _make_upload_file(content: bytes, *, filename: str = "test.bin", size: int | None = None) -> UploadFile:
    """Create a real FastAPI UploadFile backed by a BytesIO buffer."""
    buf = io.BytesIO(content)
    upload = UploadFile(file=buf, filename=filename, size=size)
    return upload


class TestReadUploadWithSizeLimit:
    """Tests for the size-checked read helper."""

    async def test_file_under_limit_succeeds(self):
        content = b"hello world"
        result = await read_upload_with_size_limit(_make_upload_file(content), max_upload_bytes=1024)
        assert result == content

    async def test_file_exactly_at_limit_succeeds(self):
        content = b"x" * 100
        result = await read_upload_with_size_limit(_make_upload_file(content), max_upload_bytes=100)
        assert result == content

    async def test_file_over_limit_raises(self):
        content = b"x" * 101
        with pytest.raises(FileTooLargeError, match="exceeds the maximum allowed upload size"):
            await read_upload_with_size_limit(_make_upload_file(content), max_upload_bytes=100)

    async def test_file_over_limit_error_has_413_status(self):
        content = b"x" * 200
        with pytest.raises(FileTooLargeError) as exc_info:
            await read_upload_with_size_limit(_make_upload_file(content), max_upload_bytes=100)
        assert exc_info.value.status_code == 413

    async def test_oversized_file_rejected_before_reading(self):
        upload = _make_upload_file(b"x" * 200)
        with pytest.raises(FileTooLargeError, match="200 bytes"):
            await read_upload_with_size_limit(upload, max_upload_bytes=100)
        assert upload.file.tell() == 0

    async def test_large_file_read(self):
        content = b"A" * (2 * 1024 * 1024 + 512 * 1024)
        result = await read_upload_with_size_limit(_make_upload_file(content), max_upload_bytes=10 * 1024 * 1024)
        assert result == content
        assert len(result) == len(content)

    async def test_empty_file_succeeds(self):
        result = await read_upload_with_size_limit(_make_upload_file(b""), max_upload_bytes=100)
        assert result == b""

    async def test_error_message_includes_sizes(self):
        content = b"x" * 2000
        with pytest.raises(FileTooLargeError, match="1000 bytes") as exc_info:
            await read_upload_with_size_limit(_make_upload_file(content), max_upload_bytes=1000)
        msg = str(exc_info.value)
        assert "2000" in msg or "exceeds" in msg


class TestPreReadUploadFile:
    """Tests for the pre-read wrapper."""

    async def test_read_returns_content(self):
        wrapper = PreReadUploadFile(b"hello", filename="test.txt", content_type="text/plain")
        assert await wrapper.read() == b"hello"

    async def test_attributes_preserved(self):
        wrapper = PreReadUploadFile(b"data", filename="doc.pdf", content_type="application/pdf")
        assert wrapper.filename == "doc.pdf"
        assert wrapper.content_type == "application/pdf"
        assert wrapper.size == 4

    async def test_is_upload_file_subclass(self):
        wrapper = PreReadUploadFile(b"content")
        assert isinstance(wrapper, UploadFile)


class TestEnforceUploadSizeLimit:
    """Tests for the size check that leaves the upload unread for streaming providers."""

    def test_file_at_limit_is_left_unread(self):
        upload = _make_upload_file(b"x" * 100)
        enforce_upload_size_limit(upload, max_upload_bytes=100)
        assert upload.file.tell() == 0

    def test_file_over_limit_raises(self):
        with pytest.raises(FileTooLargeError, match="File size 101 bytes"):
            enforce_upload_size_limit(_make_upload_file(b"x" * 101), max_upload_bytes=100)

    def test_understated_declared_size_is_ignored(self):
        with pytest.raises(FileTooLargeError, match="File size 200 bytes"):
            enforce_upload_size_limit(_make_upload_file(b"x" * 200, size=10), max_upload_bytes=100)

    async def test_partially_read_file_is_measured_in_full(self):
        """The provider streams from the start, so the whole file is what must fit the limit."""
        upload = _make_upload_file(b"x" * 150)
        await upload.read(100)
        with pytest.raises(FileTooLargeError, match="File size 150 bytes"):
            enforce_upload_size_limit(upload, max_upload_bytes=100)

    async def test_file_is_rewound_for_the_provider(self):
        upload = _make_upload_file(b"hello")
        await upload.read()
        enforce_upload_size_limit(upload, max_upload_bytes=100)
        assert await upload.read() == b"hello"
