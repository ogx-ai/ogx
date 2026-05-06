# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Responses API.

This module defines the FastAPI router for the Responses API using standard
FastAPI route decorators.
"""

import asyncio
import contextvars
import json
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Path, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ogx_api.common.responses import Order
from ogx_api.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAICompactedResponse,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
    OpenAIResponseObjectStreamError,
)
from ogx_api.router_utils import (
    ExceptionTranslatingRoute,
    create_path_dependency,
    create_query_dependency,
    standard_responses,
    try_translate_to_http_exception,
)
from ogx_api.version import OGX_API_V1

from .api import Responses
from .models import (
    CancelResponseRequest,
    CompactResponseRequest,
    CreateResponseRequest,
    DeleteResponseRequest,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    ResponseItemInclude,
    RetrieveResponseRequest,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "agents"})


def _parse_form_value(value: str) -> Any:
    """Try to parse a form value as JSON, falling back to the raw string."""
    try:
        return json.loads(value)
    except ValueError:
        return value


def _build_form_data_dict(form_data: Any) -> dict[str, Any]:
    """Build a dict from form data, collecting repeated keys into lists."""
    data: dict[str, Any] = {}
    for key, value in form_data.multi_items():
        parsed = _parse_form_value(value) if isinstance(value, str) else value
        if key in data:
            # Convert to list on second occurrence, append on subsequent
            existing = data[key]
            if isinstance(existing, list):
                existing.append(parsed)
            else:
                data[key] = [existing, parsed]
        else:
            data[key] = parsed
    return data


class FormURLEncodedRoute(ExceptionTranslatingRoute):
    """Route class that converts form-urlencoded bodies to JSON before FastAPI parses them.

    This allows Body(...) to handle both JSON and form-urlencoded requests transparently,
    preserving FastAPI's automatic OpenAPI schema generation.
    """

    def get_route_handler(self) -> Any:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            content_type = request.headers.get("content-type", "").split(";")[0].strip()
            if content_type == "application/x-www-form-urlencoded":
                form_data = await request.form()
                data = _build_form_data_dict(form_data)
                # Replace body with JSON so FastAPI's Body() parser works
                request._body = json.dumps(data).encode()
                # Update content-type so FastAPI parses as JSON
                request.scope["headers"] = [
                    (b"content-type", b"application/json") if k == b"content-type" else (k, v)
                    for k, v in request.scope["headers"]
                ]
                # Clear cached headers/form so Starlette re-reads from scope
                for attr in ("_headers", "_form"):
                    if hasattr(request, attr):
                        delattr(request, attr)
            resp: Response = await original(request)
            return resp

        return handler


def create_sse_event(data: Any) -> str:
    """Create a Server-Sent Event string from data."""
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)
    return f"data: {data}\n\n"


async def sse_generator(event_gen: AsyncIterator[Any]) -> AsyncIterator[str]:
    """Convert an async generator to SSE format.

    This function iterates over an async generator and formats each yielded
    item as a Server-Sent Event.
    """
    # Track the last sequence_number seen so that if an error occurs mid-stream,
    # the error event can continue the sequence (last seen + 1).
    sequence_number = 0
    try:
        async for item in event_gen:
            if hasattr(item, "sequence_number"):
                sequence_number = item.sequence_number
            yield create_sse_event(item)
    except asyncio.CancelledError:
        if hasattr(event_gen, "aclose"):
            await event_gen.aclose()
        raise  # Re-raise to maintain proper cancellation semantics
    except Exception as e:
        logger.exception("Error in SSE generator")
        http_exc = try_translate_to_http_exception(e)
        status_code = str(http_exc.status_code) if http_exc else "server_error"
        detail = http_exc.detail if http_exc else "Internal server error: An unexpected error occurred."
        yield create_sse_event(
            OpenAIResponseObjectStreamError(
                code=status_code,
                message=detail,
                sequence_number=sequence_number + 1,
            )
        )


# Automatically generate dependency functions from Pydantic models
get_retrieve_response_request = create_path_dependency(RetrieveResponseRequest)
get_delete_response_request = create_path_dependency(DeleteResponseRequest)
get_cancel_response_request = create_path_dependency(CancelResponseRequest)
get_list_responses_request = create_query_dependency(ListResponsesRequest)


# Manual dependency for ListResponseInputItemsRequest since it mixes Path and Query parameters
async def get_list_response_input_items_request(
    response_id: Annotated[str, Path(description="The ID of the response to retrieve input items for.")],
    after: Annotated[
        str | None,
        Query(description="An item ID to list items after, used for pagination."),
    ] = None,
    before: Annotated[
        str | None,
        Query(description="An item ID to list items before, used for pagination."),
    ] = None,
    include: Annotated[
        list[ResponseItemInclude] | None,
        Query(description="Additional fields to include in the response."),
    ] = None,
    limit: Annotated[
        int | None,
        Query(
            description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20."
        ),
    ] = 20,
    order: Annotated[Order | None, Query(description="The order to return the input items in.")] = Order.desc,
) -> ListResponseInputItemsRequest:
    return ListResponseInputItemsRequest(
        response_id=response_id,
        after=after,
        before=before,
        include=include,
        limit=limit,
        order=order,
    )


def _preserve_context_for_sse(event_gen: AsyncIterator[str]) -> AsyncIterator[str]:
    # StreamingResponse runs in a different task, losing request contextvars.
    # create_task inside context.run captures the context at task creation.
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


def create_router(impl: Responses) -> APIRouter:
    """Create a FastAPI router for the Responses API.

    Args:
        impl: The Responses implementation instance

    Returns:
        APIRouter configured for the Responses API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Responses"],
        responses=standard_responses,
        route_class=FormURLEncodedRoute,
    )

    @router.post(
        "/responses/compact",
        response_model=OpenAICompactedResponse,
        summary="Compact a conversation. [alpha]",
        description="**[alpha]** Compresses conversation history into a smaller representation while preserving context. This endpoint is in alpha and may change without notice.",
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/x-www-form-urlencoded": {},
                },
            }
        },
    )
    async def compact_openai_response(
        request: Annotated[CompactResponseRequest, Body(...)],
    ) -> OpenAICompactedResponse:
        return await impl.compact_openai_response(request)

    @router.get(
        "/responses/{response_id}",
        response_model=OpenAIResponseObject,
        summary="Get a model response.",
        description="Get a model response.",
    )
    async def get_openai_response(
        request: Annotated[RetrieveResponseRequest, Depends(get_retrieve_response_request)],
    ) -> OpenAIResponseObject:
        return await impl.get_openai_response(request)

    @router.post(
        "/responses",
        summary="Create a model response.",
        description="Create a model response.",
        status_code=200,
        response_model=None,
        responses={
            200: {
                "description": "An OpenAIResponseObject or a stream of OpenAIResponseObjectStream.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/OpenAIResponseObject"}},
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/OpenAIResponseObjectStream"}},
                },
            }
        },
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/x-www-form-urlencoded": {},
                },
            }
        },
    )
    async def create_openai_response(
        request: Annotated[CreateResponseRequest, Body(...)],
    ) -> OpenAIResponseObject | StreamingResponse:
        result = await impl.create_openai_response(request)

        # For streaming responses, wrap in StreamingResponse for HTTP requests.
        # The implementation is typed to return an `AsyncIterator` for streaming.
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                _preserve_context_for_sse(sse_generator(result)),
                media_type="text/event-stream",
            )

        return result

    @router.get(
        "/responses",
        response_model=ListOpenAIResponseObject,
        summary="List all responses.",
        description="List all responses.",
    )
    async def list_openai_responses(
        request: Annotated[ListResponsesRequest, Depends(get_list_responses_request)],
    ) -> ListOpenAIResponseObject:
        return await impl.list_openai_responses(request)

    @router.get(
        "/responses/{response_id}/input_items",
        response_model=ListOpenAIResponseInputItem,
        summary="List input items.",
        description="List input items.",
    )
    async def list_openai_response_input_items(
        request: Annotated[
            ListResponseInputItemsRequest,
            Depends(get_list_response_input_items_request),
        ],
    ) -> ListOpenAIResponseInputItem:
        return await impl.list_openai_response_input_items(request)

    @router.delete(
        "/responses/{response_id}",
        response_model=OpenAIDeleteResponseObject,
        summary="Delete a response.",
        description="Delete a response.",
    )
    async def delete_openai_response(
        request: Annotated[DeleteResponseRequest, Depends(get_delete_response_request)],
    ) -> OpenAIDeleteResponseObject:
        return await impl.delete_openai_response(request)

    @router.post(
        "/responses/{response_id}/cancel",
        response_model=OpenAIResponseObject,
        summary="Cancel a background response that is in progress.",
        description="Cancel a background response that is queued or in progress. Only responses created with background=true can be cancelled.",
        responses={
            200: {"description": "The updated response object with status 'cancelled'."},
            404: {"description": "Response not found."},
            409: {
                "description": "Conflict: Cannot cancel response (not a background response or already in terminal state)."
            },
        },
    )
    async def cancel_openai_response(
        request: Annotated[CancelResponseRequest, Depends(get_cancel_response_request)],
    ) -> OpenAIResponseObject:
        return await impl.cancel_openai_response(request)

    return router
