# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from ogx.core.request_headers import PROVIDER_DATA_VAR
from ogx.core.task import (
    RequestContext,
    activate_request_context,
    capture_request_context,
    create_detached_background_task,
)


class _CollectingExporter(SpanExporter):
    """Collects finished spans in memory for test assertions."""

    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


async def test_detached_task_runs_coroutine():
    """The helper creates a task that actually runs the coroutine to completion."""
    result = []

    async def work():
        result.append("done")

    task = create_detached_background_task(work())
    await task
    assert result == ["done"]


async def test_detached_task_clears_otel_context():
    """The task should run with an empty OTel context, not the parent's."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    captured_span = {}

    async def capture_context():
        captured_span["inner"] = trace.get_current_span()

    with tracer.start_as_current_span("parent-span"):
        parent_ctx = otel_context.get_current()
        parent_span = trace.get_current_span()

        task = create_detached_background_task(capture_context())
        await task

        assert not captured_span["inner"].is_recording()
        assert parent_span.is_recording()
        assert otel_context.get_current() == parent_ctx


async def test_detached_task_clears_provider_data():
    """The task should run with PROVIDER_DATA_VAR cleared."""
    captured = {}
    token = PROVIDER_DATA_VAR.set({"__authenticated_user": "alice"})

    async def capture_provider():
        captured["value"] = PROVIDER_DATA_VAR.get()

    try:
        task = create_detached_background_task(capture_provider())
        await task

        assert captured["value"] is None, "Background task should not inherit PROVIDER_DATA_VAR"
        assert PROVIDER_DATA_VAR.get() == {"__authenticated_user": "alice"}, "Caller's context should be unaffected"
    finally:
        PROVIDER_DATA_VAR.reset(token)


async def test_detached_task_restores_caller_context():
    """The calling coroutine's context is not affected by creating a detached task."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    token = PROVIDER_DATA_VAR.set({"__authenticated_user": "bob"})
    try:
        with tracer.start_as_current_span("parent-span"):
            otel_before = otel_context.get_current()
            provider_before = PROVIDER_DATA_VAR.get()

            create_detached_background_task(asyncio.sleep(0))

            assert otel_context.get_current() == otel_before
            assert PROVIDER_DATA_VAR.get() == provider_before
    finally:
        PROVIDER_DATA_VAR.reset(token)


async def test_detached_task_produces_independent_trace():
    """Spans created inside a detached task belong to a separate trace."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    async def background_work():
        with tracer.start_as_current_span("background-db-write"):
            await asyncio.sleep(0)

    with tracer.start_as_current_span("http-request"):
        task = create_detached_background_task(background_work())
        await task

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    request_span = span_by_name["http-request"]
    bg_span = span_by_name["background-db-write"]

    assert request_span.context.trace_id != bg_span.context.trace_id
    assert bg_span.parent is None


async def test_normal_child_task_shares_trace():
    """Contrast: a regular asyncio.create_task DOES inherit the parent trace."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    async def child_work():
        with tracer.start_as_current_span("child-span"):
            await asyncio.sleep(0)

    with tracer.start_as_current_span("parent-request"):
        task = asyncio.create_task(child_work())
        await task

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    parent_span = span_by_name["parent-request"]
    child_span = span_by_name["child-span"]

    assert parent_span.context.trace_id == child_span.context.trace_id, (
        "Regular create_task should share the parent's trace"
    )


async def test_context_through_queue_pattern():
    """End-to-end: context captured at enqueue time is correctly attached in a detached worker.

    This simulates the inference_store pattern:
    1. Request creates a span and enqueues work with captured context
    2. Worker runs in a detached (empty) context
    3. Worker attaches the captured context before processing
    4. The resulting span belongs to the original request's trace
    """
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    queue: asyncio.Queue[tuple[str, RequestContext]] = asyncio.Queue()

    async def worker():
        item, ctx = await queue.get()
        with activate_request_context(ctx):
            with tracer.start_as_current_span(f"db-write-{item}"):
                await asyncio.sleep(0)
        queue.task_done()

    token = PROVIDER_DATA_VAR.set({"user": "A"})
    try:
        with tracer.start_as_current_span("http-request-A"):
            ctx_a = capture_request_context()
            await queue.put(("A", ctx_a))
    finally:
        PROVIDER_DATA_VAR.reset(token)

    worker_task = create_detached_background_task(worker())
    await worker_task
    await queue.join()

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    request_span = span_by_name["http-request-A"]
    write_span = span_by_name["db-write-A"]

    assert request_span.context.trace_id == write_span.context.trace_id, (
        "DB write should belong to the same trace as the originating request"
    )


async def test_capture_and_activate_request_context():
    """capture_request_context snapshots both OTel and provider data; activate restores both."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    token = PROVIDER_DATA_VAR.set({"__authenticated_user": "charlie"})
    try:
        with tracer.start_as_current_span("request"):
            ctx = capture_request_context()
            request_trace_id = trace.get_current_span().get_span_context().trace_id

        assert isinstance(ctx, RequestContext)
        assert ctx.provider_data == {"__authenticated_user": "charlie"}

        # After span ends, activate context and verify OTel trace is restored
        with activate_request_context(ctx):
            with tracer.start_as_current_span("reattached-work"):
                reattached_trace_id = trace.get_current_span().get_span_context().trace_id
            assert PROVIDER_DATA_VAR.get() == {"__authenticated_user": "charlie"}

        assert request_trace_id == reattached_trace_id
    finally:
        PROVIDER_DATA_VAR.reset(token)


async def test_activate_restores_on_exit():
    """activate_request_context restores the previous context when the block exits."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")

    token = PROVIDER_DATA_VAR.set({"__authenticated_user": "outer_user"})
    try:
        with tracer.start_as_current_span("outer"):
            outer_otel = otel_context.get_current()

            inner_ctx = RequestContext(
                otel_ctx=otel_context.Context(),
                provider_data={"__authenticated_user": "inner_user"},
            )
            with activate_request_context(inner_ctx):
                assert PROVIDER_DATA_VAR.get() == {"__authenticated_user": "inner_user"}

            assert PROVIDER_DATA_VAR.get() == {"__authenticated_user": "outer_user"}
            assert otel_context.get_current() == outer_otel
    finally:
        PROVIDER_DATA_VAR.reset(token)


async def test_context_through_queue_no_cross_contamination():
    """Two requests enqueue work; each item's context is correctly propagated."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    queue: asyncio.Queue[tuple[str, RequestContext]] = asyncio.Queue()
    processed = asyncio.Event()

    async def worker():
        for _ in range(2):
            label, ctx = await queue.get()
            with activate_request_context(ctx):
                assert PROVIDER_DATA_VAR.get() == {"user": label}
                with tracer.start_as_current_span(f"db-write-{label}"):
                    await asyncio.sleep(0)
            queue.task_done()
        processed.set()

    worker_task = create_detached_background_task(worker())

    token_a = PROVIDER_DATA_VAR.set({"user": "A"})
    with tracer.start_as_current_span("request-A"):
        await queue.put(("A", capture_request_context()))
    PROVIDER_DATA_VAR.reset(token_a)

    token_b = PROVIDER_DATA_VAR.set({"user": "B"})
    with tracer.start_as_current_span("request-B"):
        await queue.put(("B", capture_request_context()))
    PROVIDER_DATA_VAR.reset(token_b)

    await processed.wait()
    await worker_task

    provider.force_flush()
    span_by_name = {s.name: s for s in exporter.spans}

    request_a = span_by_name["request-A"]
    request_b = span_by_name["request-B"]
    write_a = span_by_name["db-write-A"]
    write_b = span_by_name["db-write-B"]

    assert write_a.context.trace_id == request_a.context.trace_id
    assert write_b.context.trace_id == request_b.context.trace_id
    assert request_a.context.trace_id != request_b.context.trace_id
