# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for vector IO metrics."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.core.routers.vector_io import VectorIORouter
from ogx.telemetry.vector_io_metrics import (
    create_vector_metric_attributes,
    vector_chunks_processed_total,
    vector_deletes_total,
    vector_files_total,
    vector_insert_duration,
    vector_inserts_total,
    vector_queries_total,
    vector_retrieval_duration,
    vector_stores_total,
)


class TestVectorMetricAttributes:
    """Test metric attribute creation utility."""

    def test_create_vector_metric_attributes_all_fields(self):
        attrs = create_vector_metric_attributes(
            vector_db="vs_abc123",
            operation="chunks",
            provider="chromadb",
            status="success",
        )
        assert attrs == {
            "vector_db": "vs_abc123",
            "operation": "chunks",
            "provider": "chromadb",
            "status": "success",
        }

    def test_create_vector_metric_attributes_partial_fields(self):
        attrs = create_vector_metric_attributes(
            vector_db="vs_abc123",
            status="error",
        )
        assert attrs == {
            "vector_db": "vs_abc123",
            "status": "error",
        }
        assert "operation" not in attrs
        assert "provider" not in attrs

    def test_create_vector_metric_attributes_empty(self):
        attrs = create_vector_metric_attributes()
        assert attrs == {}


class TestVectorMetricInstruments:
    """Test that metric instruments are properly defined."""

    def test_counters_exist(self):
        for counter in [
            vector_inserts_total,
            vector_queries_total,
            vector_deletes_total,
            vector_stores_total,
            vector_files_total,
            vector_chunks_processed_total,
        ]:
            assert counter is not None
            assert hasattr(counter, "add")

    def test_histograms_exist(self):
        for histogram in [vector_insert_duration, vector_retrieval_duration]:
            assert histogram is not None
            assert hasattr(histogram, "record")

    def test_counters_can_record(self):
        attrs = create_vector_metric_attributes(
            vector_db="vs_test",
            operation="chunks",
            provider="chromadb",
            status="success",
        )
        # Should not raise
        vector_inserts_total.add(1, attrs)
        vector_queries_total.add(1, attrs)
        vector_deletes_total.add(1, attrs)
        vector_stores_total.add(1, attrs)
        vector_files_total.add(1, attrs)
        vector_chunks_processed_total.add(10, attrs)

    def test_histograms_can_record(self):
        attrs = create_vector_metric_attributes(
            vector_db="vs_test",
            provider="chromadb",
        )
        # Should not raise
        vector_insert_duration.record(1.234, attrs)
        vector_retrieval_duration.record(0.567, attrs)


class TestVectorMetricsConstants:
    """Test that metric constants follow naming conventions."""

    def test_metric_names_follow_convention(self):
        from ogx.telemetry.constants import (
            VECTOR_CHUNKS_PROCESSED_TOTAL,
            VECTOR_DELETES_TOTAL,
            VECTOR_FILES_TOTAL,
            VECTOR_INSERT_DURATION,
            VECTOR_INSERTS_TOTAL,
            VECTOR_QUERIES_TOTAL,
            VECTOR_RETRIEVAL_DURATION,
            VECTOR_STORES_TOTAL,
        )

        for name in [
            VECTOR_INSERTS_TOTAL,
            VECTOR_QUERIES_TOTAL,
            VECTOR_DELETES_TOTAL,
            VECTOR_STORES_TOTAL,
            VECTOR_FILES_TOTAL,
            VECTOR_CHUNKS_PROCESSED_TOTAL,
        ]:
            assert name.startswith("ogx.")
            assert "vector_io" in name
            assert name.endswith("_total")

        for name in [VECTOR_INSERT_DURATION, VECTOR_RETRIEVAL_DURATION]:
            assert name.startswith("ogx.")
            assert "vector_io" in name
            assert name.endswith("_seconds")


class TestVectorIORouterMetricsIntegration:
    """Test vector IO router integration with metrics."""

    def _create_mock_router(self):
        mock_routing_table = MagicMock()
        mock_routing_table.dist_registry.get_cached.return_value = MagicMock(provider_id="chromadb")
        return VectorIORouter(routing_table=mock_routing_table), mock_routing_table

    async def test_insert_chunks_records_metrics(self):
        router, mock_rt = self._create_mock_router()
        mock_rt.insert_chunks = AsyncMock(return_value=None)

        mock_request = MagicMock()
        mock_request.vector_store_id = "vs_test"
        mock_request.chunks = [MagicMock(document_id=f"doc_{i}") for i in range(5)]
        mock_request.ttl_seconds = None

        with (
            patch.object(vector_inserts_total, "add") as mock_counter,
            patch.object(vector_insert_duration, "record") as mock_duration,
            patch.object(vector_chunks_processed_total, "add") as mock_chunks,
        ):
            await router.insert_chunks(mock_request)

            mock_counter.assert_called_once()
            attrs = mock_counter.call_args[0][1]
            assert attrs["status"] == "success"
            assert attrs["vector_db"] == "vs_test"

            mock_duration.assert_called_once()
            assert mock_duration.call_args[0][0] >= 0  # duration >= 0

            mock_chunks.assert_called_once_with(5, mock_chunks.call_args[0][1])

    async def test_insert_chunks_records_error_metrics(self):
        router, mock_rt = self._create_mock_router()
        mock_rt.insert_chunks = AsyncMock(side_effect=RuntimeError("insert failed"))

        mock_request = MagicMock()
        mock_request.vector_store_id = "vs_test"
        mock_request.chunks = [MagicMock(document_id="doc_1")]
        mock_request.ttl_seconds = None

        with (
            patch.object(vector_inserts_total, "add") as mock_counter,
            patch.object(vector_insert_duration, "record"),
        ):
            with pytest.raises(RuntimeError, match="insert failed"):
                await router.insert_chunks(mock_request)

            mock_counter.assert_called_once()
            attrs = mock_counter.call_args[0][1]
            assert attrs["status"] == "error"

    async def test_insert_chunks_records_cancelled_error_metrics(self):
        router, mock_rt = self._create_mock_router()
        mock_rt.insert_chunks = AsyncMock(side_effect=asyncio.CancelledError())

        mock_request = MagicMock()
        mock_request.vector_store_id = "vs_test"
        mock_request.chunks = [MagicMock(document_id="doc_1")]
        mock_request.ttl_seconds = None

        with (
            patch.object(vector_inserts_total, "add") as mock_counter,
            patch.object(vector_insert_duration, "record"),
        ):
            with pytest.raises(asyncio.CancelledError):
                await router.insert_chunks(mock_request)

            mock_counter.assert_called_once()
            attrs = mock_counter.call_args[0][1]
            assert attrs["status"] == "error"

    async def test_query_chunks_records_metrics(self):
        router, mock_rt = self._create_mock_router()
        mock_result = MagicMock()
        mock_rt.query_chunks = AsyncMock(return_value=mock_result)

        mock_request = MagicMock()
        mock_request.vector_store_id = "vs_test"
        mock_request.query = "test query"
        mock_request.params = None

        with (
            patch.object(vector_queries_total, "add") as mock_counter,
            patch.object(vector_retrieval_duration, "record") as mock_duration,
        ):
            result = await router.query_chunks(mock_request)

            assert result == mock_result
            mock_counter.assert_called_once()
            attrs = mock_counter.call_args[0][1]
            assert attrs["status"] == "success"
            assert attrs["operation"] == "query"
            assert attrs["search_mode"] == "vector"

            mock_duration.assert_called_once()

    async def test_search_vector_store_records_metrics(self):
        router, mock_rt = self._create_mock_router()
        mock_result = MagicMock()
        mock_rt.openai_search_vector_store = AsyncMock(return_value=mock_result)

        mock_request = MagicMock()
        mock_request.query = "test search"
        mock_request.rewrite_query = False
        mock_request.search_mode = "hybrid"
        mock_request.model_copy.return_value = mock_request

        with (
            patch.object(vector_queries_total, "add") as mock_counter,
            patch.object(vector_retrieval_duration, "record") as mock_duration,
        ):
            result = await router.openai_search_vector_store("vs_test", mock_request)

            assert result == mock_result
            mock_counter.assert_called_once()
            attrs = mock_counter.call_args[0][1]
            assert attrs["status"] == "success"
            assert attrs["operation"] == "search"
            assert attrs["search_mode"] == "hybrid"

            mock_duration.assert_called_once()

    async def test_delete_vector_store_records_metrics(self):
        router, mock_rt = self._create_mock_router()
        mock_result = MagicMock()
        mock_rt.openai_delete_vector_store = AsyncMock(return_value=mock_result)

        with patch.object(vector_deletes_total, "add") as mock_counter:
            result = await router.openai_delete_vector_store("vs_test")

            assert result == mock_result
            mock_counter.assert_called_once()
            attrs = mock_counter.call_args[0][1]
            assert attrs["vector_db"] == "vs_test"
            assert attrs["operation"] == "store"
