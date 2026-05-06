# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Google Interactions API.

This module defines the FastAPI router for the /v1/interactions endpoint,
serving the Google Interactions API format.
"""

import asyncio
import contextvars
import json
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ogx_api.common.errors import ModelNotFoundError
from ogx_api.router_utils import standard_responses
from ogx_api.version import OGX_API_V1ALPHA

from .api import Interactions
from .models import (
    GoogleCreateInteractionRequest,
    GoogleErrorResponse,
    GoogleInteractionResponse,
    _GoogleErrorDetail,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "interactions"})


def _create_google_sse_event(event_type: str, data: Any) -> str:
    """Create a Google-format SSE event with named event type.

    Google SSE format: event: <type>\ndata: <json>\n\n
    """
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)
    return f"event: {event_type}\ndata: {data}\n\n"


async def _google_sse_generator(event_gen: AsyncIterator) -> AsyncIterator[str]:
    """Convert an async generator of Google stream events to SSE format."""
    try:
        async for event in event_gen:
            event_type = event.event_type if hasattr(event, "event_type") else "unknown"
            yield _create_google_sse_event(event_type, event)
    except asyncio.CancelledError:
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise
    except Exception as e:
        logger.exception("Error in Google SSE generator")
        error_resp = GoogleErrorResponse(
            error=_GoogleErrorDetail(code=500, message=str(e)),
        )
        yield _create_google_sse_event("error", error_resp)


def _preserve_context_for_sse(event_gen):
    """Preserve request context for SSE streaming.

    StreamingResponse runs in a different task, losing request contextvars.
    This wrapper captures and restores the context.
    """
    context = contextvars.copy_context()

    async def wrapper():
        try:
            while True:
                try:
                    task = context.run(asyncio.create_task, event_gen.__anext__())
                    item = await task
                except StopAsyncIteration:
                    break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            if hasattr(event_gen, "aclose"):
                await event_gen.aclose()
            raise

    return wrapper()


def _google_error_response(status_code: int, message: str) -> JSONResponse:
    """Create a Google-format error JSONResponse."""
    body = GoogleErrorResponse(
        error=_GoogleErrorDetail(code=status_code, message=message),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def create_router(impl: Interactions) -> APIRouter:
    """Create a FastAPI router for the Google Interactions API.

    Args:
        impl: The Interactions implementation instance

    Returns:
        APIRouter configured for the Interactions API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1ALPHA}",
        tags=["Interactions"],
        responses=standard_responses,
    )

    @router.post(
        "/interactions",
        summary="Create an interaction.",
        description="Create an interaction using the Google Interactions API format.",
        status_code=200,
        response_model=GoogleInteractionResponse,
        responses={
            200: {
                "description": "A GoogleInteractionResponse or a stream of Google SSE events.",
                "content": {
                    "text/event-stream": {},
                },
            },
        },
    )
    async def create_interaction(
        raw_request: Request,
        params: Annotated[GoogleCreateInteractionRequest, Body(...)],
    ) -> Response:
        try:
            result = await impl.create_interaction(params)
        except NotImplementedError as e:
            return _google_error_response(501, str(e))
        except ModelNotFoundError as e:
            return _google_error_response(404, str(e))
        except ValueError as e:
            return _google_error_response(400, str(e))
        except HTTPException as e:
            return _google_error_response(e.status_code, e.detail)
        except Exception:
            logger.exception("Failed to create interaction")
            return _google_error_response(500, "Internal server error")

        if getattr(result, "_raw_sse", False):
            # Raw SSE passthrough — forward bytes directly, no context wrapping
            # needed since the stream doesn't access request contextvars
            return StreamingResponse(cast(AsyncIterator[str], result), media_type="text/event-stream")
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(_google_sse_generator(cast(AsyncIterator[Any], result))),
                media_type="text/event-stream",
            )

        return JSONResponse(
            content=result.model_dump(exclude_none=True),
        )

    return router
