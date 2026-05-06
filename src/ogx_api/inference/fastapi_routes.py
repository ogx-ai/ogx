# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Inference API.

This module defines the FastAPI router for the Inference API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

import asyncio
import contextvars
import json
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ogx_api.common.errors import OpenAIErrorResponse
from ogx_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1, OGX_API_V1ALPHA

from .api import Inference
from .models import (
    GetChatCompletionRequest,
    ListChatCompletionsRequest,
    ListOpenAIChatCompletionResponse,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankRequest,
    RerankResponse,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "inference"})


def _create_sse_event(data: Any) -> str:
    """Create a Server-Sent Event string from data."""
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)
    return f"data: {data}\n\n"


async def _sse_generator(event_gen: AsyncIterator[Any], context: str = "inference") -> AsyncIterator[str]:
    """Convert an async generator to SSE format."""
    try:
        async for item in event_gen:
            yield _create_sse_event(item)
    except asyncio.CancelledError:
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise
    except Exception as e:
        logger.exception(f"Error in SSE generator ({context})")
        exc = _http_exception_from_sse_error(e)
        yield _create_sse_event(OpenAIErrorResponse.from_message(exc.detail, code=str(exc.status_code)).to_dict())


def _http_exception_from_value_error(exc: ValueError) -> HTTPException:
    """Convert a ValueError to an HTTPException."""
    detail = str(exc) or "Invalid value"
    return HTTPException(status_code=400, detail=detail)


def _http_exception_from_sse_error(exc: Exception) -> HTTPException:
    """Convert an exception to an HTTPException."""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return _http_exception_from_value_error(exc)
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return HTTPException(status_code=status_code, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal server error: An unexpected error occurred.")


def _preserve_context_for_sse(event_gen: AsyncIterator[str]) -> AsyncIterator[str]:
    """Preserve request context for SSE streaming.

    StreamingResponse runs in a different task, losing request contextvars.
    This wrapper captures and restores the context.
    """
    context = contextvars.copy_context()

    async def wrapper() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    task: asyncio.Task[str] = context.run(asyncio.create_task, event_gen.__anext__())  # type: ignore[arg-type]
                    item = await task
                except StopAsyncIteration:
                    break
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            if hasattr(event_gen, "aclose"):
                await event_gen.aclose()
            raise

    return wrapper()


# Automatically generate dependency functions from Pydantic models
# This ensures the models are the single source of truth for descriptions
get_list_chat_completions_request = create_query_dependency(ListChatCompletionsRequest)
get_chat_completion_request = create_path_dependency(GetChatCompletionRequest)


def create_router(impl: Inference) -> APIRouter:
    """Create a FastAPI router for the Inference API.

    Args:
        impl: The Inference implementation instance

    Returns:
        APIRouter configured for the Inference API
    """
    # Use no prefix - specify full paths for each route to support both v1 and v1alpha endpoints
    router = APIRouter(
        tags=["Inference"],
        responses=standard_responses,
    )

    @router.post(
        f"/{OGX_API_V1}/chat/completions",
        response_model=None,  # Dynamic response: non-streaming (JSON) or streaming (SSE)
        summary="Create chat completions.",
        description="Generate an OpenAI-compatible chat completion for the given messages using the specified model.",
        responses={
            200: {
                "description": "An OpenAIChatCompletion. When streaming, returns Server-Sent Events (SSE) with OpenAIChatCompletionChunk objects.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/OpenAIChatCompletion"}},
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/OpenAIChatCompletionChunk"}},
                },
            },
        },
    )
    async def openai_chat_completion(
        params: Annotated[OpenAIChatCompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAIChatCompletion | StreamingResponse:
        result = await impl.openai_chat_completion(params)
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(_sse_generator(result, context="chat_completion")),
                media_type="text/event-stream",
            )
        return result

    @router.get(
        f"/{OGX_API_V1}/chat/completions",
        response_model=ListOpenAIChatCompletionResponse,
        summary="List chat completions.",
        description="List chat completions.",
        responses={
            200: {"description": "A ListOpenAIChatCompletionResponse."},
        },
    )
    async def list_chat_completions(
        request: Annotated[ListChatCompletionsRequest, Depends(get_list_chat_completions_request)],
    ) -> ListOpenAIChatCompletionResponse:
        return await impl.list_chat_completions(request)

    @router.get(
        f"/{OGX_API_V1}/chat/completions/{{completion_id}}",
        response_model=OpenAICompletionWithInputMessages,
        summary="Get chat completion.",
        description="Describe a chat completion by its ID.",
        responses={
            200: {"description": "A OpenAICompletionWithInputMessages."},
        },
    )
    async def get_chat_completion(
        request: Annotated[GetChatCompletionRequest, Depends(get_chat_completion_request)],
    ) -> OpenAICompletionWithInputMessages:
        return await impl.get_chat_completion(request)

    @router.post(
        f"/{OGX_API_V1}/completions",
        response_model=None,  # Dynamic response: non-streaming (JSON) or streaming (SSE)
        summary="Create completion.",
        description="Generate an OpenAI-compatible completion for the given prompt using the specified model.",
        responses={
            200: {
                "description": "An OpenAICompletion. When streaming, returns Server-Sent Events (SSE) with OpenAICompletion chunks.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/OpenAICompletion"}},
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/OpenAICompletion"}},
                },
            },
        },
    )
    async def openai_completion(
        params: Annotated[OpenAICompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAICompletion | StreamingResponse:
        result = await impl.openai_completion(params)
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(_sse_generator(result, context="completion")),
                media_type="text/event-stream",
            )
        return result

    @router.post(
        f"/{OGX_API_V1}/embeddings",
        response_model=OpenAIEmbeddingsResponse,
        summary="Create embeddings.",
        description="Generate OpenAI-compatible embeddings for the given input using the specified model.",
        responses={
            200: {"description": "An OpenAIEmbeddingsResponse containing the embeddings."},
        },
    )
    async def openai_embeddings(
        params: Annotated[OpenAIEmbeddingsRequestWithExtraBody, Body(...)],
    ) -> OpenAIEmbeddingsResponse:
        return await impl.openai_embeddings(params)

    @router.post(
        f"/{OGX_API_V1ALPHA}/inference/rerank",
        response_model=RerankResponse,
        summary="Rerank documents based on relevance to a query.",
        description="Rerank a list of documents based on their relevance to a query.",
        responses={
            200: {"description": "RerankResponse with indices sorted by relevance score (descending)."},
        },
    )
    async def rerank(
        request: Annotated[RerankRequest, Body(...)],
    ) -> RerankResponse:
        return await impl.rerank(request)

    return router
