# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from pydantic import SecretStr

from ogx.providers.remote.file_processor.unstructured_api.config import UnstructuredApiFileProcessorConfig
from ogx.providers.remote.file_processor.unstructured_api.unstructured_api import UnstructuredApiFileProcessor
from ogx_api.file_processors import ProcessFileRequest

# Mock Unstructured API response
MOCK_ELEMENTS = [
    {
        "type": "Title",
        "text": "Introduction to Machine Learning",
        "metadata": {"page_number": 1, "filename": "test.pdf"},
    },
    {
        "type": "NarrativeText",
        "text": "Machine learning is a subset of artificial intelligence.",
        "metadata": {"page_number": 1, "filename": "test.pdf"},
    },
    {
        "type": "ListItem",
        "text": "Supervised Learning",
        "metadata": {"page_number": 2, "filename": "test.pdf"},
    },
    {
        "type": "Table",
        "text": "Column1 | Column2\nData1 | Data2",
        "metadata": {"page_number": 3, "filename": "test.pdf"},
    },
    {
        "type": "NarrativeText",
        "text": "",  # Empty element - should be skipped
        "metadata": {"page_number": 3, "filename": "test.pdf"},
    },
]


class TestUnstructuredApiFileProcessor:
    @pytest.fixture
    def config(self) -> UnstructuredApiFileProcessorConfig:
        return UnstructuredApiFileProcessorConfig(
            api_key=SecretStr("test-api-key-123"),
            default_chunk_size_tokens=800,
        )

    @pytest.fixture
    def files_api(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def processor(
        self, config: UnstructuredApiFileProcessorConfig, files_api: AsyncMock
    ) -> UnstructuredApiFileProcessor:
        return UnstructuredApiFileProcessor(config, files_api=files_api)

    @pytest.fixture
    def upload_file(self) -> UploadFile:
        return UploadFile(file=io.BytesIO(b"%PDF-fake-content"), filename="test.pdf")

    # -- input validation --

    async def test_rejects_no_file_and_no_file_id(self, processor: UnstructuredApiFileProcessor):
        request = ProcessFileRequest()
        with pytest.raises(ValueError, match="Either file or file_id must be provided"):
            await processor.process_file(request)

    async def test_rejects_both_file_and_file_id(
        self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile
    ):
        request = ProcessFileRequest(file_id="file-123")
        with pytest.raises(ValueError, match="Cannot provide both file and file_id"):
            await processor.process_file(request, file=upload_file)

    # -- process file with mock API --

    async def test_process_file_success(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()

        # Mock the Unstructured API client
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        # Verify API was called
        mock_client.general.partition_async.assert_called_once()

        # Verify response structure
        assert len(response.chunks) == 4  # 5 elements, but 1 is empty and skipped
        assert response.metadata["processor"] == "unstructured-api"
        assert response.metadata["extraction_method"] == "unstructured-api"
        assert "processing_time_ms" in response.metadata
        assert response.metadata["total_elements"] == 5
        assert response.metadata["file_size_bytes"] == len(b"%PDF-fake-content")

    async def test_element_types_preserved(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        # Check element types are preserved
        element_types = [chunk.metadata["element_type"] for chunk in response.chunks]
        assert "Title" in element_types
        assert "NarrativeText" in element_types
        assert "ListItem" in element_types
        assert "Table" in element_types

    async def test_empty_elements_skipped(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        # Should skip the empty element
        assert len(response.chunks) == 4
        assert all(chunk.content.strip() for chunk in response.chunks)

    # -- chunk metadata mapping --

    async def test_chunk_metadata_fields(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS[:1]  # Just first element

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        chunk = response.chunks[0]

        # Verify metadata fields
        uuid.UUID(chunk.metadata["document_id"])
        assert chunk.metadata["filename"] == "test.pdf"
        assert chunk.metadata["element_type"] == "Title"
        assert chunk.metadata["element_index"] == 0
        assert chunk.metadata["page_number"] == 1

        # Verify chunk_metadata
        assert chunk.chunk_id == chunk.chunk_metadata.chunk_id
        assert chunk.chunk_metadata.document_id == chunk.metadata["document_id"]
        assert chunk.chunk_metadata.source == "test.pdf"
        assert chunk.chunk_metadata.content_token_count > 0

    async def test_chunk_id_uniqueness(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        # All chunk IDs should be unique
        ids = [c.chunk_id for c in response.chunks]
        assert len(ids) == len(set(ids))

    async def test_page_numbers_preserved(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        # Check page numbers
        page_numbers = [chunk.metadata.get("page_number") for chunk in response.chunks]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers

    async def test_token_count_calculated(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS[:1]

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request, file=upload_file)

        chunk = response.chunks[0]
        # "Introduction to Machine Learning" = 4 tokens (whitespace split)
        assert chunk.chunk_metadata.content_token_count == 4

    # -- file_id path --

    async def test_process_file_via_file_id(self, config: UnstructuredApiFileProcessorConfig):
        files_api = AsyncMock()
        files_api.openai_retrieve_file.return_value = SimpleNamespace(filename="report.pdf")
        files_api.openai_retrieve_file_content.return_value = SimpleNamespace(body=b"%PDF-fake")

        processor = UnstructuredApiFileProcessor(config, files_api=files_api)
        request = ProcessFileRequest(file_id="file-abc")

        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS[:1]

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            response = await processor.process_file(request)

        files_api.openai_retrieve_file.assert_awaited_once()
        files_api.openai_retrieve_file_content.assert_awaited_once()
        assert response.chunks[0].metadata["filename"] == "report.pdf"
        assert response.chunks[0].metadata["file_id"] == "file-abc"
        assert response.chunks[0].metadata["document_id"] == "file-abc"

    # -- API key authentication --

    async def test_api_key_used(self, processor: UnstructuredApiFileProcessor, upload_file: UploadFile):
        request = ProcessFileRequest()
        mock_response = MagicMock()
        mock_response.elements = MOCK_ELEMENTS[:1]

        with patch(
            "ogx.providers.remote.file_processor.unstructured_api.unstructured_api.UnstructuredClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.general.partition_async = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await processor.process_file(request, file=upload_file)

        # Verify client was initialized with API key
        mock_client_class.assert_called_once_with(api_key_auth="test-api-key-123")


class TestUnstructuredApiFileProcessorConfig:
    def test_default_values(self):
        config = UnstructuredApiFileProcessorConfig(api_key=SecretStr("test-key"))
        assert config.api_key.get_secret_value() == "test-key"
        assert config.default_chunk_size_tokens >= 100

    def test_sample_run_config(self):
        sample = UnstructuredApiFileProcessorConfig.sample_run_config()
        assert "api_key" in sample
        assert "${env.UNSTRUCTURED_API_KEY}" in sample["api_key"]
