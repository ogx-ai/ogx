# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Anthropic Messages API.

This module defines the FastAPI router for the /v1/messages endpoint,
serving the Anthropic Messages API format.
"""

import asyncio
import contextvars
import json
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ogx_api.common.errors import ModelNotFoundError
from ogx_api.router_utils import standard_responses
from ogx_api.version import OGX_API_V1

from .api import Messages
from .models import (
    ANTHROPIC_VERSION,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicErrorResponse,
    AnthropicMessageResponse,
    _AnthropicErrorDetail,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "messages"})


def _create_anthropic_sse_event(event_type: str, data: Any) -> str:
    """Create an Anthropic-format SSE event with named event type.

    Anthropic SSE format: event: <type>\ndata: <json>\n\n
    """
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)
    return f"event: {event_type}\ndata: {data}\n\n"


async def _anthropic_sse_generator(event_gen: AsyncIterator) -> AsyncIterator[str]:
    """Convert an async generator of Anthropic stream events to SSE format."""
    try:
        async for event in event_gen:
            event_type = event.type if hasattr(event, "type") else "unknown"
            yield _create_anthropic_sse_event(event_type, event)
    except asyncio.CancelledError:
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise
    except Exception as e:
        logger.exception("Error in Anthropic SSE generator")
        error_resp = AnthropicErrorResponse(
            error=_AnthropicErrorDetail(type="api_error", message=str(e)),
        )
        yield _create_anthropic_sse_event("error", error_resp)


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


def _anthropic_error_response(status_code: int, message: str) -> JSONResponse:
    """Create an Anthropic-format error JSONResponse."""
    error_type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }
    error_type = error_type_map.get(status_code, "api_error")
    body = AnthropicErrorResponse(
        error=_AnthropicErrorDetail(type=error_type, message=message),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def create_router(impl: Messages) -> APIRouter:
    """Create a FastAPI router for the Anthropic Messages API.

    Args:
        impl: The Messages implementation instance

    Returns:
        APIRouter configured for the Messages API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Messages"],
        responses=standard_responses,
    )

    @router.post(
        "/messages",
        summary="Create a message.",
        description="Create a message using the Anthropic Messages API format.",
        status_code=200,
        response_model=AnthropicMessageResponse,
        responses={
            200: {
                "description": "An AnthropicMessageResponse or a stream of Anthropic SSE events.",
                "content": {
                    "text/event-stream": {},
                },
            },
        },
    )
    async def create_message(
        raw_request: Request,
        params: Annotated[AnthropicCreateMessageRequest, Body(...)],
    ) -> Response:
        try:
            result = await impl.create_message(params)
        except NotImplementedError as e:
            return _anthropic_error_response(501, str(e))
        except ModelNotFoundError as e:
            return _anthropic_error_response(404, str(e))
        except ValueError as e:
            return _anthropic_error_response(400, str(e))
        except HTTPException as e:
            return _anthropic_error_response(e.status_code, e.detail)
        except Exception:
            logger.exception("Failed to create message")
            return _anthropic_error_response(500, "Internal server error")

        response_headers = {"anthropic-version": ANTHROPIC_VERSION}

        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(_anthropic_sse_generator(result)),
                media_type="text/event-stream",
                headers=response_headers,
            )

        return JSONResponse(
            content=result.model_dump(exclude_none=True),
            headers=response_headers,
        )

    @router.post(
        "/messages/count_tokens",
        response_model=AnthropicCountTokensResponse,
        summary="Count tokens in a message.",
        description="Count the number of tokens in a message request.",
        responses={
            200: {"description": "Token count for the request."},
        },
    )
    async def count_message_tokens(
        params: Annotated[AnthropicCountTokensRequest, Body(...)],
    ) -> Response:
        try:
            result = await impl.count_message_tokens(params)
        except NotImplementedError as e:
            return _anthropic_error_response(501, str(e))
        except Exception:
            logger.exception("Failed to count message tokens")
            return _anthropic_error_response(500, "Internal server error")

        return JSONResponse(
            content=result.model_dump(),
            headers={"anthropic-version": ANTHROPIC_VERSION},
        )

    return router
