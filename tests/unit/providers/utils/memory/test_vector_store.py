# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.utils.memory.vector_store import (
    content_from_data_and_mime_type,
    validate_tiktoken_encoding,
)
from ogx_api import URL, RAGDocument


def test_validate_tiktoken_encoding_succeeds():
    """validate_tiktoken_encoding does not raise when tiktoken is available."""
    validate_tiktoken_encoding("cl100k_base")


def test_validate_tiktoken_encoding_raises_on_failure():
    """validate_tiktoken_encoding raises RuntimeError with operator context on failure."""
    import ogx.providers.utils.memory.vector_store as vs_module

    with patch.object(vs_module, "_get_encoding", side_effect=RuntimeError("encoding unavailable")):
        with pytest.raises(RuntimeError, match="encoding unavailable"):
            validate_tiktoken_encoding("cl100k_base")


def test_content_from_data_and_mime_type_success_utf8():
    """Test successful decoding with UTF-8 encoding."""
    data = "Hello World! 🌍".encode()
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "utf-8"}

        result = content_from_data_and_mime_type(data, mime_type)

        mock_detect.assert_called_once_with(data)
        assert result == "Hello World! 🌍"


def test_content_from_data_and_mime_type_error_win1252():
    """Test fallback to UTF-8 when Windows-1252 encoding detection fails."""
    data = "Hello World! 🌍".encode()
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "Windows-1252"}

        result = content_from_data_and_mime_type(data, mime_type)

        assert result == "Hello World! 🌍"
        mock_detect.assert_called_once_with(data)


def test_content_from_data_and_mime_type_both_encodings_fail():
    """Test that exceptions are raised when both primary and UTF-8 encodings fail."""
    # Create invalid byte sequence that fails with both encodings
    data = b"\xff\xfe\x00\x8f"  # Invalid UTF-8 sequence
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "windows-1252"}

        # Should raise an exception instead of returning empty string
        with pytest.raises(UnicodeDecodeError):
            content_from_data_and_mime_type(data, mime_type)


async def test_memory_tool_error_handling():
    """Test that memory tool handles various failures gracefully without crashing."""
    from ogx.providers.inline.tool_runtime.file_search.config import FileSearchToolRuntimeConfig
    from ogx.providers.inline.tool_runtime.file_search.file_search import FileSearchToolRuntimeImpl

    config = FileSearchToolRuntimeConfig()
    memory_tool = FileSearchToolRuntimeImpl(
        config=config,
        vector_io_api=AsyncMock(),
        inference_api=AsyncMock(),
        files_api=AsyncMock(),
    )

    docs = [
        RAGDocument(document_id="good_doc", content="Good content", metadata={}),
        RAGDocument(document_id="bad_url_doc", content=URL(uri="https://bad.url"), metadata={}),
        RAGDocument(document_id="another_good_doc", content="Another good content", metadata={}),
    ]

    mock_file1 = MagicMock()
    mock_file1.id = "file_good1"
    mock_file2 = MagicMock()
    mock_file2.id = "file_good2"
    memory_tool.files_api.openai_upload_file.side_effect = [mock_file1, mock_file2]

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.get.side_effect = Exception("Bad URL")
        mock_client.return_value.__aenter__.return_value = mock_instance

        # won't raise exception despite one document failing
        await memory_tool.insert(docs, "vector_store_123")

    # processed 2 documents successfully, skipped 1
    assert memory_tool.files_api.openai_upload_file.call_count == 2
    assert memory_tool.vector_io_api.openai_attach_file_to_vector_store.call_count == 2
