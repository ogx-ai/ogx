# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""The /v1/messages and /v1/messages/count_tokens route handlers, not just translate_exception
in isolation: a route with its own try/except swallows an exception before the server's global
exception handler (which does call translate_exception) is ever reached, so a provider error
only actually reaches the client with its real status if the route itself preserves it (#6661).
"""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from ogx.core.server.fastapi_router_registry import get_router_routes
from ogx.providers.remote.inference.anthropic.anthropic import AnthropicAPIError
from ogx_api.common.errors import ModelNotFoundError
from ogx_api.messages.api import Messages
from ogx_api.messages.fastapi_routes import create_router
from ogx_api.messages.models import AnthropicCountTokensRequest, AnthropicCreateMessageRequest, AnthropicMessage

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _create_message_endpoint():
    impl = AsyncMock(spec=Messages)
    router = create_router(impl)
    endpoint = next(r.endpoint for r in get_router_routes(router) if r.path == "/v1/messages")
    return impl, endpoint


def _count_tokens_endpoint():
    impl = AsyncMock(spec=Messages)
    router = create_router(impl)
    endpoint = next(r.endpoint for r in get_router_routes(router) if r.path == "/v1/messages/count_tokens")
    return impl, endpoint


_MESSAGES = [AnthropicMessage(role="user", content="hi")]
_MESSAGE_PARAMS = AnthropicCreateMessageRequest(model="claude-haiku-4-5", messages=_MESSAGES, max_tokens=16)
_COUNT_TOKENS_PARAMS = AnthropicCountTokensRequest(model="claude-haiku-4-5", messages=_MESSAGES)


async def test_create_message_preserves_anthropic_api_error_status():
    """The bug this test guards: AnthropicAPIError matched no except clause, so a 401/429 from
    Anthropic reached the client as a generic 500 instead of its real status."""
    impl, create_message = _create_message_endpoint()
    impl.create_message.side_effect = AnthropicAPIError(429, "rate limited")

    response = await create_message(raw_request=None, params=_MESSAGE_PARAMS)

    assert response.status_code == 429
    assert json.loads(response.body)["error"]["message"] == "rate limited"


async def test_count_message_tokens_preserves_anthropic_api_error_status():
    """count_tokens had no HTTPException branch at all before this fix, so any provider error
    -- not just AnthropicAPIError -- reached the client as a generic 500."""
    impl, count_message_tokens = _count_tokens_endpoint()
    impl.count_message_tokens.side_effect = AnthropicAPIError(401, "invalid x-api-key")

    response = await count_message_tokens(params=_COUNT_TOKENS_PARAMS)

    assert response.status_code == 401
    assert json.loads(response.body)["error"]["message"] == "invalid x-api-key"


async def test_create_message_preserves_generic_http_exception_status():
    """Confirms the new fallback doesn't regress a bare HTTPException (already handled by its
    own except clause, but try_translate_to_http_exception would also recognize it)."""
    impl, create_message = _create_message_endpoint()
    impl.create_message.side_effect = HTTPException(status_code=403, detail="forbidden")

    response = await create_message(raw_request=None, params=_MESSAGE_PARAMS)

    assert response.status_code == 403


async def test_create_message_still_maps_model_not_found_to_404():
    impl, create_message = _create_message_endpoint()
    impl.create_message.side_effect = ModelNotFoundError("no-such-model")

    response = await create_message(raw_request=None, params=_MESSAGE_PARAMS)

    assert response.status_code == 404


async def test_create_message_still_falls_back_to_500_for_an_unrecognized_exception():
    impl, create_message = _create_message_endpoint()
    impl.create_message.side_effect = RuntimeError("boom")

    response = await create_message(raw_request=None, params=_MESSAGE_PARAMS)

    assert response.status_code == 500


async def test_count_message_tokens_still_falls_back_to_500_for_an_unrecognized_exception():
    impl, count_message_tokens = _count_tokens_endpoint()
    impl.count_message_tokens.side_effect = RuntimeError("boom")

    response = await count_message_tokens(params=_COUNT_TOKENS_PARAMS)

    assert response.status_code == 500
