# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import UploadFile
from pydantic import SecretStr

from ogx.providers.remote.file_processor.docling_serve.config import DoclingServeFileProcessorConfig
from ogx.providers.remote.file_processor.docling_serve.docling_serve import DoclingServeFileProcessor
from ogx_api.file_processors import ProcessFileRequest
from ogx_api.vector_io import (
    VectorStoreChunkingStrategyAuto,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)


def _make_httpx_response(json_body: dict, status_code: int = 200) -> httpx.Response:
    """Build a minimal httpx.Response from a JSON dict."""
    return httpx.Response(
        status_code=status_code,
        json=json_body,
        request=httpx.Request("POST", "http://test"),
    )


CONVERT_RESPONSE = {
    "document": {
        "md_content": "# Hello World\n\nThis is a test document with some content.",
    },
}

CHUNK_RESPONSE = {
    "chunks": [
        {"text": "First chunk of text.", "meta": {"headings": ["Introduction"]}},
        {"text": "Second chunk of text.", "meta": {}},
        {"text": "Third chunk of text.", "meta": {"headings": ["Conclusion"]}},
    ],
}


class TestDoclingServeFileProcessor:
    @pytest.fixture
    def config(self) -> DoclingServeFileProcessorConfig:
        return DoclingServeFileProcessorConfig(
            base_url="http://localhost:5001/v1",
            default_chunk_size_tokens=512,
        )

    @pytest.fixture
    def config_with_api_key(self) -> DoclingServeFileProcessorConfig:
        return DoclingServeFileProcessorConfig(
            base_url="http://localhost:5001/v1",
            api_key=SecretStr("test-secret-key"),
        )

    @pytest.fixture
    def files_api(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def processor(self, config: DoclingServeFileProcessorConfig, files_api: AsyncMock) -> DoclingServeFileProcessor:
        return DoclingServeFileProcessor(config, files_api=files_api)

    @pytest.fixture
    def upload_file(self) -> UploadFile:
        return UploadFile(file=io.BytesIO(b"%PDF-fake-content"), filename="test.pdf")

    # -- input validation --

    async def test_rejects_no_file_and_no_file_id(self, processor: DoclingServeFileProcessor):
        request = ProcessFileRequest()
        with pytest.raises(ValueError, match="Either file or file_id must be provided"):
            await processor.process_file(request)

    async def test_rejects_both_file_and_file_id(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(file_id="file-123")
        with pytest.raises(ValueError, match="Cannot provide both file and file_id"):
            await processor.process_file(request, file=upload_file)

    # -- convert (no chunking) --

    async def test_process_file_no_chunking(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = _make_httpx_response(CONVERT_RESPONSE)

        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            response = await processor.process_file(request, file=upload_file)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "/convert/file" in call_kwargs.args[0]
        assert call_kwargs.kwargs["files"]["files"][0] == "test.pdf"

        assert len(response.chunks) == 1
        assert response.chunks[0].content == CONVERT_RESPONSE["document"]["md_content"]
        assert response.metadata["processor"] == "docling-serve"
        assert response.metadata["extraction_method"] == "docling-serve"
        assert "processing_time_ms" in response.metadata
        assert response.metadata["file_size_bytes"] == len(b"%PDF-fake-content")

    async def test_process_file_no_chunking_empty_content(
        self, processor: DoclingServeFileProcessor, upload_file: UploadFile
    ):
        request = ProcessFileRequest()
        empty_response = {"document": {"md_content": "   "}}

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(empty_response)):
            response = await processor.process_file(request, file=upload_file)

        assert len(response.chunks) == 0
        assert response.metadata["processor"] == "docling-serve"

    # -- chunk (with chunking strategy) --

    async def test_process_file_auto_chunking(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyAuto())
        mock_response = _make_httpx_response(CHUNK_RESPONSE)

        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            response = await processor.process_file(request, file=upload_file)

        call_kwargs = mock_post.call_args
        assert "/chunk/hybrid/file" in call_kwargs.args[0]
        assert call_kwargs.kwargs["data"]["chunking_max_tokens"] == "512"

        assert len(response.chunks) == 3
        assert response.chunks[0].content == "First chunk of text."
        assert response.chunks[1].content == "Second chunk of text."
        assert response.chunks[2].content == "Third chunk of text."

    async def test_process_file_static_chunking(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        static_config = VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=256)
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyStatic(static=static_config))
        mock_response = _make_httpx_response(CHUNK_RESPONSE)

        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            response = await processor.process_file(request, file=upload_file)

        call_kwargs = mock_post.call_args
        assert "/chunk/hybrid/file" in call_kwargs.args[0]
        assert call_kwargs.kwargs["data"]["chunking_max_tokens"] == "256"
        assert len(response.chunks) == 3

    async def test_chunking_empty_response(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyAuto())

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response({"chunks": []})):
            response = await processor.process_file(request, file=upload_file)

        assert len(response.chunks) == 0

    async def test_chunking_skips_blank_chunks(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyAuto())
        body = {"chunks": [{"text": "real text", "meta": {}}, {"text": "   ", "meta": {}}, {"text": "", "meta": {}}]}

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(body)):
            response = await processor.process_file(request, file=upload_file)

        assert len(response.chunks) == 1
        assert response.chunks[0].content == "real text"

    # -- chunk metadata mapping --

    async def test_chunk_metadata_fields(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CONVERT_RESPONSE)):
            response = await processor.process_file(request, file=upload_file)

        chunk = response.chunks[0]
        uuid.UUID(chunk.metadata["document_id"])
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.chunk_id == chunk.chunk_metadata.chunk_id
        assert chunk.chunk_metadata.document_id == chunk.metadata["document_id"]
        assert chunk.chunk_metadata.source == "test.pdf"
        assert chunk.chunk_metadata.content_token_count > 0

    async def test_chunk_id_uniqueness(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyAuto())

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CHUNK_RESPONSE)):
            response = await processor.process_file(request, file=upload_file)

        ids = [c.chunk_id for c in response.chunks]
        assert len(ids) == len(set(ids))

    async def test_headings_propagated(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyAuto())

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CHUNK_RESPONSE)):
            response = await processor.process_file(request, file=upload_file)

        assert response.chunks[0].metadata["headings"] == ["Introduction"]
        assert "headings" not in response.chunks[1].metadata
        assert response.chunks[2].metadata["headings"] == ["Conclusion"]

    async def test_chunk_window_set(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest(chunking_strategy=VectorStoreChunkingStrategyAuto())

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CHUNK_RESPONSE)):
            response = await processor.process_file(request, file=upload_file)

        for i, chunk in enumerate(response.chunks):
            assert chunk.chunk_metadata.chunk_window == str(i)

    # -- file_id path --

    async def test_process_file_via_file_id(self, config: DoclingServeFileProcessorConfig):
        files_api = AsyncMock()
        files_api.openai_retrieve_file.return_value = SimpleNamespace(filename="report.pdf")
        files_api.openai_retrieve_file_content.return_value = SimpleNamespace(body=b"%PDF-fake")

        processor = DoclingServeFileProcessor(config, files_api=files_api)
        request = ProcessFileRequest(file_id="file-abc")

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CONVERT_RESPONSE)):
            response = await processor.process_file(request)

        files_api.openai_retrieve_file.assert_awaited_once()
        files_api.openai_retrieve_file_content.assert_awaited_once()
        assert response.chunks[0].metadata["filename"] == "report.pdf"
        assert response.chunks[0].metadata["file_id"] == "file-abc"
        assert response.chunks[0].metadata["document_id"] == "file-abc"

    # -- api key / headers --

    async def test_api_key_header_sent(
        self, config_with_api_key: DoclingServeFileProcessorConfig, files_api: AsyncMock
    ):
        processor = DoclingServeFileProcessor(config_with_api_key, files_api=files_api)
        request = ProcessFileRequest()
        upload = UploadFile(file=io.BytesIO(b"data"), filename="doc.pdf")

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CONVERT_RESPONSE)) as mock_post:
            await processor.process_file(request, file=upload)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Api-Key"] == "test-secret-key"

    async def test_no_api_key_header_when_unset(self, processor: DoclingServeFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CONVERT_RESPONSE)) as mock_post:
            await processor.process_file(request, file=upload_file)

        headers = mock_post.call_args.kwargs["headers"]
        assert "X-Api-Key" not in headers

    # -- mime type --

    async def test_mime_type_for_known_extensions(self, processor: DoclingServeFileProcessor):
        request = ProcessFileRequest()

        for filename, expected_mime in [
            ("test.pdf", "application/pdf"),
            ("test.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("test.html", "text/html"),
            ("test.png", "image/png"),
        ]:
            upload = UploadFile(file=io.BytesIO(b"data"), filename=filename)
            with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CONVERT_RESPONSE)) as mock_post:
                await processor.process_file(request, file=upload)

            sent_files = mock_post.call_args.kwargs["files"]["files"]
            assert sent_files[2] == expected_mime

    async def test_mime_type_fallback_for_unknown_extension(self, processor: DoclingServeFileProcessor):
        request = ProcessFileRequest()
        upload = UploadFile(file=io.BytesIO(b"data"), filename="test.xyz")

        with patch("httpx.AsyncClient.post", return_value=_make_httpx_response(CONVERT_RESPONSE)) as mock_post:
            await processor.process_file(request, file=upload)

        sent_files = mock_post.call_args.kwargs["files"]["files"]
        assert sent_files[2] == "application/octet-stream"


class TestDoclingServeFileProcessorConfig:
    def test_default_values(self):
        config = DoclingServeFileProcessorConfig()
        assert config.base_url == "http://localhost:5001/v1"
        assert config.api_key is None
        assert config.default_chunk_size_tokens >= 100

    def test_sample_run_config(self):
        sample = DoclingServeFileProcessorConfig.sample_run_config()
        assert "base_url" in sample
        assert "api_key" in sample
