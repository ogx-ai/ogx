# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io

import pytest
from fastapi import UploadFile

from ogx_api.common.errors import FileTooLargeError
from ogx_api.common.upload_safety import (
    PreReadUploadFile,
    read_upload_with_size_limit,
)


def _make_upload_file(content: bytes, *, filename: str = "test.bin", size: int | None = None) -> UploadFile:
    """Create a real FastAPI UploadFile backed by a BytesIO buffer."""
    buf = io.BytesIO(content)
    upload = UploadFile(file=buf, filename=filename, size=size)
    return upload


class _NoSizeUploadFile:
    """File-like object without .size that supports chunked read(n).

    Simulates a non-UploadFile object that supports the read(n) interface
    but does not expose a .size attribute (e.g. a future library client variant).
    """

    def __init__(self, content: bytes):
        self.filename = "test.bin"
        self._buf = io.BytesIO(content)

    async def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)


class _UnchunkedUploadFile:
    """File-like object whose read() takes no args, like LibraryClientUploadFile.

    Backed by BytesIO so repeated calls hit EOF. Passing a size argument
    raises TypeError, exercising the fallback path.
    """

    def __init__(self, content: bytes):
        self.filename = "test.bin"
        self._buf = io.BytesIO(content)

    async def read(self) -> bytes:
        return self._buf.read()


def _make_no_size_upload(content: bytes) -> _NoSizeUploadFile:
    return _NoSizeUploadFile(content)


def _make_unchunked_upload(content: bytes) -> _UnchunkedUploadFile:
    return _UnchunkedUploadFile(content)


class TestReadUploadWithSizeLimit:
    """Tests for the bounded-read helper."""

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

    async def test_precheck_rejects_when_size_known(self):
        """When file.size is set, reject before reading any bytes."""
        content = b"x" * 200
        upload = _make_upload_file(content, size=200)
        with pytest.raises(FileTooLargeError, match="200 bytes"):
            await read_upload_with_size_limit(upload, max_upload_bytes=100)

    async def test_precheck_not_triggered_when_size_none(self):
        """When file.size is None, fall through to chunked read."""
        content = b"x" * 200
        upload = _make_upload_file(content, size=None)
        with pytest.raises(FileTooLargeError):
            await read_upload_with_size_limit(upload, max_upload_bytes=100)

    async def test_large_file_chunked_read(self):
        """Verify multi-chunk reads work correctly for files larger than one chunk."""
        # 2.5 MB — will require multiple 1 MB chunk reads
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

    async def test_no_size_attr_under_limit(self):
        """File-like objects without .size use chunked reading normally."""
        upload = _make_no_size_upload(b"hello")
        result = await read_upload_with_size_limit(upload, max_upload_bytes=1024)
        assert result == b"hello"

    async def test_no_size_attr_over_limit(self):
        """File-like objects without .size are still rejected via chunked read."""
        upload = _make_no_size_upload(b"x" * 200)
        with pytest.raises(FileTooLargeError):
            await read_upload_with_size_limit(upload, max_upload_bytes=100)

    async def test_no_size_attr_large_file_chunked_read(self):
        """Multi-chunk reads work for file-like objects without .size."""
        # 2.5 MB — will require multiple 1 MB chunk reads
        content = b"B" * (2 * 1024 * 1024 + 512 * 1024)
        upload = _make_no_size_upload(content)
        result = await read_upload_with_size_limit(upload, max_upload_bytes=10 * 1024 * 1024)
        assert result == content
        assert len(result) == len(content)

    async def test_no_chunked_read_support_under_limit(self):
        """Objects whose read() takes no args fall back to single read."""
        upload = _make_unchunked_upload(b"hello")
        result = await read_upload_with_size_limit(upload, max_upload_bytes=1024)
        assert result == b"hello"

    async def test_no_chunked_read_support_over_limit(self):
        """Objects whose read() takes no args still get rejected with clear message."""
        upload = _make_unchunked_upload(b"x" * 200)
        with pytest.raises(FileTooLargeError, match=r"200 bytes.*exceeds.*100 bytes"):
            await read_upload_with_size_limit(upload, max_upload_bytes=100)


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
