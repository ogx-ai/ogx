# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx_api import (
    VectorStoreChunkingStrategyAuto,
)
from ogx_api.vector_io.models import OpenAIAttachFileRequest


def _make_store_info():
    """Build a minimal in-memory vector store dict matching the mixin's expectations."""
    return {
        "file_ids": [],
        "file_counts": {"total": 0, "completed": 0, "cancelled": 0, "failed": 0, "in_progress": 0},
        "metadata": {},
    }


class MockVectorStoreMixin(OpenAIVectorStoreMixin):
    """Mock implementation of OpenAIVectorStoreMixin for testing."""

    def __init__(self, inference_api, files_api, kvstore=None, file_processor_api=None):
        super().__init__(
            inference_api=inference_api,
            files_api=files_api,
            kvstore=kvstore,
            file_processor_api=file_processor_api,
        )

    async def register_vector_store(self, vector_store):
        pass

    async def unregister_vector_store(self, vector_store_id):
        pass

    async def insert_chunks(self, request):
        pass

    async def query_chunks(self, request):
        pass

    async def delete_chunks(self, request):
        pass


class TestOpenAIVectorStoreMixin:
    """Unit tests for OpenAIVectorStoreMixin."""

    @pytest.fixture
    def mock_files_api(self):
        mock = AsyncMock()
        mock.openai_retrieve_file = AsyncMock()
        mock.openai_retrieve_file.return_value = MagicMock(filename="test.pdf")
        return mock

    @pytest.fixture
    def mock_inference_api(self):
        return AsyncMock()

    @pytest.fixture
    def mock_kvstore(self):
        kv = AsyncMock()
        kv.set = AsyncMock()
        kv.get = AsyncMock(return_value=None)
        return kv

    async def test_missing_file_processor_api_returns_failed_status(
        self, mock_inference_api, mock_files_api, mock_kvstore
    ):
        """Test that missing file_processor_api marks the file as failed with a clear error."""
        mixin = MockVectorStoreMixin(
            inference_api=mock_inference_api,
            files_api=mock_files_api,
            kvstore=mock_kvstore,
            file_processor_api=None,
        )

        vector_store_id = "test_vector_store"
        file_id = "test_file_id"
        mixin.openai_vector_stores[vector_store_id] = _make_store_info()

        result = await mixin.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            request=OpenAIAttachFileRequest(
                file_id=file_id,
                chunking_strategy=VectorStoreChunkingStrategyAuto(),
            ),
        )

        assert result.status == "failed"
        assert result.last_error is not None
        assert "FileProcessor API is required" in result.last_error.message

    async def test_file_processor_api_configured_succeeds(self, mock_inference_api, mock_files_api, mock_kvstore):
        """Test that with file_processor_api configured, processing proceeds past the check."""
        mock_file_processor_api = AsyncMock()
        mock_file_processor_api.process_file = AsyncMock()
        mock_file_processor_api.process_file.return_value = MagicMock(chunks=[], metadata={"processor": "pypdf"})

        mixin = MockVectorStoreMixin(
            inference_api=mock_inference_api,
            files_api=mock_files_api,
            kvstore=mock_kvstore,
            file_processor_api=mock_file_processor_api,
        )

        vector_store_id = "test_vector_store"
        file_id = "test_file_id"
        mixin.openai_vector_stores[vector_store_id] = _make_store_info()

        result = await mixin.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            request=OpenAIAttachFileRequest(
                file_id=file_id,
                chunking_strategy=VectorStoreChunkingStrategyAuto(),
            ),
        )

        # Should not fail with the file_processor_api error
        if result.last_error:
            assert "FileProcessor API is required" not in result.last_error.message
