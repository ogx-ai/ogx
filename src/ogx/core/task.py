# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import Coroutine
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import context as otel_context

from ogx.core.request_headers import PROVIDER_DATA_VAR


@dataclass
class RequestContext:
    """Snapshot of request-scoped state for propagation through background queues.

    Background workers are long-lived asyncio tasks whose contextvars are frozen
    at creation time.  Capturing both the OTel trace context and the provider /
    auth data at *enqueue* time and re-activating them per work-item ensures:

    * Each DB write is attributed to the correct request trace (OTel).
    * Each DB write is stamped with the correct user identity (PROVIDER_DATA_VAR).
    """

    otel_ctx: otel_context.Context
    provider_data: Any


def capture_request_context() -> RequestContext:
    """Snapshot the current request-scoped context for later use in a worker."""
    return RequestContext(
        otel_ctx=otel_context.get_current(),
        provider_data=PROVIDER_DATA_VAR.get(),
    )


@contextmanager
def activate_request_context(ctx: RequestContext):
    """Temporarily restore a previously captured request context.

    Use this in worker loops that run with a detached (empty) context to
    attribute work back to the originating request.
    """
    otel_token = otel_context.attach(ctx.otel_ctx)
    provider_token = PROVIDER_DATA_VAR.set(ctx.provider_data)
    try:
        yield
    finally:
        PROVIDER_DATA_VAR.reset(provider_token)
        otel_context.detach(otel_token)


def create_detached_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create an asyncio task that does not inherit request-scoped context.

    asyncio.create_task copies all contextvars at creation time, which causes
    long-lived background workers to permanently inherit the spawning request's
    OTel trace and auth identity.  This helper temporarily clears both before
    creating the task, then immediately restores them so the caller is unaffected.
    """
    otel_token = otel_context.attach(otel_context.Context())
    provider_token = PROVIDER_DATA_VAR.set(None)
    try:
        task = asyncio.create_task(coro)
    finally:
        PROVIDER_DATA_VAR.reset(provider_token)
        otel_context.detach(otel_token)
    return task
