# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from ogx.core.storage.datatypes import InferenceStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIUserMessageParam,
    Order,
)


class _CollectingExporter(SpanExporter):
    """Collects finished spans in memory for test assertions."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


@pytest.fixture(autouse=True)
def setup_backends(tmp_path):
    """Register SQL store backends for testing."""
    db_path = str(tmp_path / "test.db")
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=db_path)})


def create_test_chat_completion(
    completion_id: str, created_timestamp: int, model: str = "test-model"
) -> OpenAIChatCompletion:
    """Helper to create a test chat completion."""
    return OpenAIChatCompletion(
        id=completion_id,
        created=created_timestamp,
        model=model,
        object="chat.completion",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content=f"Response for {completion_id}",
                ),
                finish_reason="stop",
            )
        ],
    )


async def test_inference_store_pagination_basic():
    """Test basic pagination functionality."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data with different timestamps
    base_time = int(time.time())
    test_data = [
        ("zebra-task", base_time + 1),
        ("apple-job", base_time + 2),
        ("moon-work", base_time + 3),
        ("banana-run", base_time + 4),
        ("car-exec", base_time + 5),
    ]

    # Store test chat completions
    for completion_id, timestamp in test_data:
        completion = create_test_chat_completion(completion_id, timestamp)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test 1: First page with limit=2, descending order (default)
    result = await store.list_chat_completions(limit=2, order=Order.desc)
    assert len(result.data) == 2
    assert result.data[0].id == "car-exec"  # Most recent first
    assert result.data[1].id == "banana-run"
    assert result.has_more is True
    assert result.last_id == "banana-run"

    # Test 2: Second page using 'after' parameter
    result2 = await store.list_chat_completions(after="banana-run", limit=2, order=Order.desc)
    assert len(result2.data) == 2
    assert result2.data[0].id == "moon-work"
    assert result2.data[1].id == "apple-job"
    assert result2.has_more is True

    # Test 3: Final page
    result3 = await store.list_chat_completions(after="apple-job", limit=2, order=Order.desc)
    assert len(result3.data) == 1
    assert result3.data[0].id == "zebra-task"
    assert result3.has_more is False


async def test_inference_store_pagination_ascending():
    """Test pagination with ascending order."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data
    base_time = int(time.time())
    test_data = [
        ("delta-item", base_time + 1),
        ("charlie-task", base_time + 2),
        ("alpha-work", base_time + 3),
    ]

    # Store test chat completions
    for completion_id, timestamp in test_data:
        completion = create_test_chat_completion(completion_id, timestamp)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test ascending order pagination
    result = await store.list_chat_completions(limit=1, order=Order.asc)
    assert len(result.data) == 1
    assert result.data[0].id == "delta-item"  # Oldest first
    assert result.has_more is True

    # Second page with ascending order
    result2 = await store.list_chat_completions(after="delta-item", limit=1, order=Order.asc)
    assert len(result2.data) == 1
    assert result2.data[0].id == "charlie-task"
    assert result2.has_more is True


async def test_inference_store_pagination_with_model_filter():
    """Test pagination combined with model filtering."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data with different models
    base_time = int(time.time())
    test_data = [
        ("xyz-task", base_time + 1, "model-a"),
        ("def-work", base_time + 2, "model-b"),
        ("pqr-job", base_time + 3, "model-a"),
        ("abc-run", base_time + 4, "model-b"),
    ]

    # Store test chat completions
    for completion_id, timestamp, model in test_data:
        completion = create_test_chat_completion(completion_id, timestamp, model)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test pagination with model filter
    result = await store.list_chat_completions(limit=1, model="model-a", order=Order.desc)
    assert len(result.data) == 1
    assert result.data[0].id == "pqr-job"  # Most recent model-a
    assert result.data[0].model == "model-a"
    assert result.has_more is True

    # Second page with model filter
    result2 = await store.list_chat_completions(after="pqr-job", limit=1, model="model-a", order=Order.desc)
    assert len(result2.data) == 1
    assert result2.data[0].id == "xyz-task"
    assert result2.data[0].model == "model-a"
    assert result2.has_more is False


async def test_inference_store_pagination_invalid_after():
    """Test error handling for invalid 'after' parameter."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Try to paginate with non-existent ID
    with pytest.raises(ValueError, match="Record with id='non-existent' not found in table 'chat_completions'"):
        await store.list_chat_completions(after="non-existent", limit=2)


async def test_inference_store_pagination_no_limit():
    """Test pagination behavior when no limit is specified."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data
    base_time = int(time.time())
    test_data = [
        ("omega-first", base_time + 1),
        ("beta-second", base_time + 2),
    ]

    # Store test chat completions
    for completion_id, timestamp in test_data:
        completion = create_test_chat_completion(completion_id, timestamp)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test without limit
    result = await store.list_chat_completions(order=Order.desc)
    assert len(result.data) == 2
    assert result.data[0].id == "beta-second"  # Most recent first
    assert result.data[1].id == "omega-first"
    assert result.has_more is False


async def test_inference_store_custom_table_name():
    """Test that the table_name from config is respected."""
    custom_table_name = "custom_inference_store"
    reference = InferenceStoreReference(backend="sql_default", table_name=custom_table_name)
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create and store a test chat completion
    base_time = int(time.time())
    completion = create_test_chat_completion("custom-table-test", base_time)
    input_messages = [OpenAIUserMessageParam(role="user", content="Test custom table")]
    await store.store_chat_completion(completion, input_messages)
    await store.flush()

    # Verify we can retrieve the completion
    result = await store.get_chat_completion("custom-table-test")
    assert result.id == "custom-table-test"
    assert result.model == "test-model"

    # Verify listing works
    list_result = await store.list_chat_completions()
    assert len(list_result.data) == 1
    assert list_result.data[0].id == "custom-table-test"

    # Verify the error message uses the custom table name
    with pytest.raises(ValueError, match=f"Record with id='non-existent' not found in table '{custom_table_name}'"):
        await store.list_chat_completions(after="non-existent", limit=2)


async def test_otel_traces_not_leaked_across_requests():
    """Two concurrent requests produce clean, separate OTel traces.

    Reproduces the bug observed in Jaeger traces where background worker tasks
    permanently inherited the first request's OTel context. This caused all
    subsequent DB writes from other requests to appear under that trace,
    inflating it from 5s to 62s with 334 unrelated INSERT operations.

    The fix captures OTel context at enqueue time and attaches it per-item
    in the worker loop, so each DB write is attributed to its originating request.
    """
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    reference = InferenceStoreReference(
        backend="sql_default",
        table_name="otel_test_completions",
        num_writers=1,
    )
    store = InferenceStore(reference, policy=[])
    await store.initialize()
    store.enable_write_queue = True

    original_write = store._write_chat_completion

    async def instrumented_write(completion, messages):
        with tracer.start_as_current_span(f"db-write-{completion.id}"):
            await original_write(completion, messages)

    store._write_chat_completion = instrumented_write

    base_time = int(time.time())
    completion_a = create_test_chat_completion("completion-A", base_time + 1)
    completion_b = create_test_chat_completion("completion-B", base_time + 2)
    messages_a = [OpenAIUserMessageParam(role="user", content="request A")]
    messages_b = [OpenAIUserMessageParam(role="user", content="request B")]

    # Simulate two API requests arriving in sequence (as the InferenceRouter does:
    # asyncio.create_task(store.store_chat_completion(...)) inside a request span).
    with tracer.start_as_current_span("request-A"):
        task_a = asyncio.create_task(store.store_chat_completion(completion_a, messages_a))
    await task_a

    with tracer.start_as_current_span("request-B"):
        task_b = asyncio.create_task(store.store_chat_completion(completion_b, messages_b))
    await task_b

    await store.flush()
    await store.shutdown()

    provider.force_flush()
    spans_by_name = {}
    for s in exporter.spans:
        spans_by_name[s.name] = s

    request_a_trace = spans_by_name["request-A"].context.trace_id
    request_b_trace = spans_by_name["request-B"].context.trace_id
    write_a_trace = spans_by_name["db-write-completion-A"].context.trace_id
    write_b_trace = spans_by_name["db-write-completion-B"].context.trace_id

    assert request_a_trace != request_b_trace, "Requests should have distinct trace IDs"

    assert write_a_trace == request_a_trace, (
        f"DB write for completion-A should be in request-A's trace, "
        f"got trace {write_a_trace:#x} expected {request_a_trace:#x}"
    )
    assert write_b_trace == request_b_trace, (
        f"DB write for completion-B should be in request-B's trace, "
        f"got trace {write_b_trace:#x} expected {request_b_trace:#x}"
    )


async def test_otel_worker_does_not_inherit_first_request_trace():
    """Workers start with a detached context and don't permanently adopt any request's trace.

    Before the fix, the worker task was created via loop.create_task() inside
    the first request's span context, permanently binding all future work to
    that trace. This test verifies that worker-internal operations (like queue
    polling) don't produce spans under any request's trace.
    """
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    reference = InferenceStoreReference(
        backend="sql_default",
        table_name="otel_worker_test",
        num_writers=1,
    )
    store = InferenceStore(reference, policy=[])
    await store.initialize()
    store.enable_write_queue = True

    original_write = store._write_chat_completion

    async def instrumented_write(completion, messages):
        with tracer.start_as_current_span(f"db-write-{completion.id}"):
            await original_write(completion, messages)

    store._write_chat_completion = instrumented_write

    base_time = int(time.time())

    # First request spawns the worker (this is where the old bug lived:
    # the worker permanently inherited request-1's trace context)
    with tracer.start_as_current_span("request-1-spawns-worker"):
        first_request_trace = trace.get_current_span().get_span_context().trace_id
        completion_1 = create_test_chat_completion("comp-1", base_time + 1)
        task = asyncio.create_task(
            store.store_chat_completion(
                completion_1,
                [OpenAIUserMessageParam(role="user", content="first")],
            )
        )
    await task
    await store.flush()

    # Second request enqueues work; worker is already running
    with tracer.start_as_current_span("request-2"):
        second_request_trace = trace.get_current_span().get_span_context().trace_id
        completion_2 = create_test_chat_completion("comp-2", base_time + 2)
        task = asyncio.create_task(
            store.store_chat_completion(
                completion_2,
                [OpenAIUserMessageParam(role="user", content="second")],
            )
        )
    await task
    await store.flush()

    # Third request (no trace context at all)
    completion_3 = create_test_chat_completion("comp-3", base_time + 3)
    await store.store_chat_completion(
        completion_3,
        [OpenAIUserMessageParam(role="user", content="third")],
    )
    await store.flush()
    await store.shutdown()

    provider.force_flush()
    spans_by_name = {s.name: s for s in exporter.spans}

    # Write 1 should be in request-1's trace
    assert spans_by_name["db-write-comp-1"].context.trace_id == first_request_trace

    # Write 2 should be in request-2's trace, NOT request-1's
    assert spans_by_name["db-write-comp-2"].context.trace_id == second_request_trace
    assert spans_by_name["db-write-comp-2"].context.trace_id != first_request_trace, (
        "BUG REPRODUCED: write for request-2 leaked into request-1's trace"
    )

    # Write 3 (no request context) should be in its own independent trace
    write_3_trace = spans_by_name["db-write-comp-3"].context.trace_id
    assert write_3_trace != first_request_trace
    assert write_3_trace != second_request_trace
