# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenTelemetry metrics for ogx vector IO operations.

This module provides centralized metric definitions for tracking:
- Vector insert/query/delete operations (counts, duration)
- Vector store and file lifecycle events
- Chunk processing volumes

All metrics follow OpenTelemetry semantic conventions and use the ogx prefix
for consistent naming across the telemetry stack.
"""

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram

from .constants import (
    VECTOR_CHUNKS_PROCESSED_TOTAL,
    VECTOR_DELETES_TOTAL,
    VECTOR_FILES_TOTAL,
    VECTOR_INSERT_DURATION,
    VECTOR_INSERTS_TOTAL,
    VECTOR_QUERIES_TOTAL,
    VECTOR_RETRIEVAL_DURATION,
    VECTOR_STORES_TOTAL,
)

# Get or create meter for ogx.vector_io
meter = metrics.get_meter("ogx.vector_io", version="1.0.0")

# Operation counters
vector_inserts_total: Counter = meter.create_counter(
    name=VECTOR_INSERTS_TOTAL,
    description="Total number of vector insert operations",
    unit="1",
)

vector_queries_total: Counter = meter.create_counter(
    name=VECTOR_QUERIES_TOTAL,
    description="Total number of vector query/search operations",
    unit="1",
)

vector_deletes_total: Counter = meter.create_counter(
    name=VECTOR_DELETES_TOTAL,
    description="Total number of vector delete operations",
    unit="1",
)

vector_stores_total: Counter = meter.create_counter(
    name=VECTOR_STORES_TOTAL,
    description="Total number of vector stores created",
    unit="1",
)

vector_files_total: Counter = meter.create_counter(
    name=VECTOR_FILES_TOTAL,
    description="Total number of files attached to vector stores",
    unit="1",
)

vector_chunks_processed_total: Counter = meter.create_counter(
    name=VECTOR_CHUNKS_PROCESSED_TOTAL,
    description="Total number of chunks processed across all insert operations",
    unit="1",
)

# Duration histograms with sub-second bucket boundaries for API latency
_DURATION_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]

vector_insert_duration: Histogram = meter.create_histogram(
    name=VECTOR_INSERT_DURATION,
    description="Duration of vector insert operations",
    unit="s",
    explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
)

vector_retrieval_duration: Histogram = meter.create_histogram(
    name=VECTOR_RETRIEVAL_DURATION,
    description="Duration of vector retrieval operations",
    unit="s",
    explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
)


def create_vector_metric_attributes(
    vector_db: str | None = None,
    operation: str | None = None,
    provider: str | None = None,
    status: str | None = None,
    search_mode: str | None = None,
) -> dict[str, str]:
    """Create a consistent attribute dictionary for vector IO metrics.

    Args:
        vector_db: Vector store identifier (e.g., "vs_abc123")
        operation: Operation type (e.g., "chunks", "search", "delete")
        provider: Provider ID (e.g., "chromadb", "faiss")
        status: Request outcome ("success", "error")
        search_mode: Search mode used (e.g., "vector", "keyword", "hybrid")

    Returns:
        Dictionary of attributes with non-None values
    """
    attributes: dict[str, str] = {}

    if vector_db is not None:
        attributes["vector_db"] = vector_db
    if operation is not None:
        attributes["operation"] = operation
    if provider is not None:
        attributes["provider"] = provider
    if status is not None:
        attributes["status"] = status
    if search_mode is not None:
        attributes["search_mode"] = search_mode

    return attributes
