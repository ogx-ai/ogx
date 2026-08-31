# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Google Interactions API.

This module defines the FastAPI router for the /v1/interactions endpoint,
serving the Google Interactions API format.
"""

import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast

from fastapi import APIRouter, Body, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ogx_api.common.errors import ModelNotFoundError
from ogx_api.router_utils import standard_responses
from ogx_api.utils import _preserve_context_for_sse, create_sse_event_with_type, sse_stream
from ogx_api.version import OGX_API_V1ALPHA

from .api import Interactions
from .models import (
    GoogleCreateInteractionRequest,
    GoogleErrorResponse,
    GoogleInteractionResponse,
    _GoogleErrorDetail,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "interactions"})


def _format_google_sse_event(event: Any) -> str:
    """Format a Google stream event as a named SSE event."""
    event_type = event.event_type if hasattr(event, "event_type") else "unknown"
    return create_sse_event_with_type(event_type, event)


def _format_google_sse_error_event(e: Exception) -> str:
    """Log and format an SSE stream error as a Google error event."""
    logger.exception("Error in Google SSE generator")
    error_resp = GoogleErrorResponse(
        error=_GoogleErrorDetail(code=500, message=str(e)),
    )
    return create_sse_event_with_type("error", error_resp)


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
        if isinstance(result, JSONResponse):
            # Raw JSON passthrough from provider — forward as-is
            return result
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(
                    sse_stream(
                        cast(AsyncIterator[Any], result),
                        _format_google_sse_event,
                        _format_google_sse_error_event,
                    )
                ),
                media_type="text/event-stream",
            )

        return JSONResponse(
            content=result.model_dump(exclude_none=True),
        )

    return router
