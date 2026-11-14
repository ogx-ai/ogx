# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""TEST_CONTEXT must be derivable from the request's provider-data header in
library_client mode too, the same way ProviderDataMiddleware derives it in server mode
(#6668). _route_call_in_process (REST-style in-process calls) and
AsyncOGXAsLibraryClient.request() (the openai-SDK-style transport) are the two in-process
entry points that bypass that middleware entirely, so each stamps the active test ID into
the header itself and derives TEST_CONTEXT from it via the same
sync_test_context_from_provider_data() the middleware uses.
"""

import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.responses import StreamingResponse
from ogx_client import AsyncStream

from ogx.core.library_client import AsyncOGXAsLibraryClient, _route_call_in_process
from ogx.core.request_headers import PROVIDER_DATA_VAR
from ogx.core.server.routes import RouteAuthInfo, RouteImpls
from ogx.core.testing_context import (
    get_test_context,
    reset_test_context,
    set_test_context,
    sync_test_context_from_provider_data,
)

pytestmark = pytest.mark.usefixtures("_ogx_test_inference_mode")


@pytest.fixture
def _ogx_test_inference_mode(monkeypatch):
    """sync_test_context_from_provider_data() is a no-op unless test mode is active."""
    monkeypatch.setenv("OGX_TEST_INFERENCE_MODE", "replay")


@pytest.fixture
def active_test_context():
    token = set_test_context("tests/unit/some_test.py::test_it")
    yield "tests/unit/some_test.py::test_it"
    reset_test_context(token)


def _make_route_impls(handler) -> RouteImpls:
    return {"post": {r"^/v1/test$": (handler, "/v1/test", RouteAuthInfo())}}


async def _call_in_process(handler, *, provider_data=None, async_streaming=False):
    return await _route_call_in_process(
        method="POST",
        url="http://localhost/v1/test",
        header_params=None,
        body=None,
        post_params=None,
        route_impls=_make_route_impls(handler),
        provider_data=provider_data,
        sanitize_headers=lambda headers: dict(headers or {}),
        convert_body=lambda _func, body, **_kwargs: body,
        async_streaming=async_streaming,
    )


class TestSyncTestContextFromProviderData:
    """The function both entry points call to derive TEST_CONTEXT must work for either
    stack mode, since it's now shared between the server middleware and the in-process path."""

    @pytest.mark.parametrize("stack_config_type", ["library_client", "server"])
    def test_derives_test_id_regardless_of_stack_config_type(self, monkeypatch, stack_config_type):
        monkeypatch.setenv("OGX_TEST_STACK_CONFIG_TYPE", stack_config_type)
        token = PROVIDER_DATA_VAR.set({"__test_id": "abc"})
        try:
            result_token = sync_test_context_from_provider_data()
            assert result_token is not None
            assert get_test_context() == "abc"
        finally:
            reset_test_context(result_token)
            PROVIDER_DATA_VAR.reset(token)

    def test_still_a_noop_outside_test_mode(self, monkeypatch):
        monkeypatch.delenv("OGX_TEST_INFERENCE_MODE", raising=False)
        token = PROVIDER_DATA_VAR.set({"__test_id": "abc"})
        try:
            assert sync_test_context_from_provider_data() is None
        finally:
            PROVIDER_DATA_VAR.reset(token)

    def test_noop_without_a_test_id_in_provider_data(self):
        token = PROVIDER_DATA_VAR.set({"some_other_key": "value"})
        try:
            assert sync_test_context_from_provider_data() is None
        finally:
            PROVIDER_DATA_VAR.reset(token)


class TestRouteCallInProcessDerivesTestContext:
    """_route_call_in_process is the shared implementation behind both the sync and async
    REST-style library clients (OGXAsLibraryClient / AsyncOGXAsLibraryClient.call_api)."""

    async def test_endpoint_sees_the_active_test_context(self, active_test_context):
        async def endpoint():
            return {"test_id": get_test_context()}

        response = await _call_in_process(endpoint)

        assert json.loads(response.response.content)["test_id"] == active_test_context

    async def test_stamps_test_id_alongside_other_provider_data(self, active_test_context):
        async def endpoint():
            return {"provider_data": PROVIDER_DATA_VAR.get()}

        response = await _call_in_process(endpoint, provider_data={"other_key": "value"})

        provider_data = json.loads(response.response.content)["provider_data"]
        assert provider_data == {"other_key": "value", "__test_id": active_test_context}

    async def test_context_is_restored_after_the_call(self, active_test_context):
        async def endpoint():
            return {}

        await _call_in_process(endpoint)

        assert get_test_context() == active_test_context

    async def test_noop_without_an_active_test_context(self):
        """No test running: the header must not gain a test ID, matching production usage."""

        async def endpoint():
            return {"test_id": get_test_context(), "provider_data": PROVIDER_DATA_VAR.get()}

        response = await _call_in_process(endpoint)

        result = json.loads(response.response.content)
        assert result["test_id"] is None
        assert result["provider_data"] == {}

    async def test_streaming_preserves_test_context_across_iteration(self):
        """The stamping with-block exits as soon as the streaming response is built, before
        the caller ever iterates it, so the captured value must survive to iteration time
        even if the ambient test context has since changed to something else."""
        token = set_test_context("original-test")
        try:

            async def endpoint() -> StreamingResponse:
                async def gen() -> AsyncGenerator[str, None]:
                    yield json.dumps({"test_id": get_test_context()})

                return StreamingResponse(gen(), media_type="text/event-stream")

            response = await _call_in_process(endpoint, async_streaming=True)
        finally:
            reset_test_context(token)

        other_token = set_test_context("a-different-test")
        try:
            payload = b"".join([chunk async for chunk in response.response.stream]).decode()
        finally:
            reset_test_context(other_token)

        assert json.loads(payload)["test_id"] == "original-test"


class TestAsyncLibraryClientRequestDerivesTestContext:
    """AsyncOGXAsLibraryClient.request() is the second in-process entry point: the transport
    used when an openai.AsyncOpenAI()-style client is pointed at the library client directly,
    bypassing _route_call_in_process entirely."""

    @staticmethod
    def _make_client(route_impls: RouteImpls) -> AsyncOGXAsLibraryClient:
        client = object.__new__(AsyncOGXAsLibraryClient)
        client.route_impls = route_impls
        client.provider_data = None
        return client

    @staticmethod
    def _options(**overrides: Any) -> SimpleNamespace:
        defaults: dict[str, Any] = {"method": "POST", "url": "/v1/test", "params": None, "json_data": {}, "headers": {}}
        return SimpleNamespace(**{**defaults, **overrides})

    async def test_non_streaming_endpoint_sees_the_active_test_context(self, active_test_context):
        async def endpoint():
            return {"test_id": get_test_context()}

        client = self._make_client(_make_route_impls(endpoint))

        response = await client.request(dict, self._options(), stream=False)

        assert response["test_id"] == active_test_context

    async def test_non_streaming_noop_without_an_active_test_context(self):
        async def endpoint():
            return {"test_id": get_test_context()}

        client = self._make_client(_make_route_impls(endpoint))

        response = await client.request(dict, self._options(), stream=False)

        assert response["test_id"] is None

    async def test_streaming_preserves_test_context_across_iteration(self):
        token = set_test_context("original-test")
        try:

            async def endpoint():
                async def gen() -> AsyncGenerator[dict, None]:
                    yield {"test_id": get_test_context()}

                return gen()

            client = self._make_client(_make_route_impls(endpoint))

            response = await client.request(dict, self._options(), stream=True, stream_cls=AsyncStream[dict])
        finally:
            reset_test_context(token)

        other_token = set_test_context("a-different-test")
        try:
            chunks = [chunk async for chunk in response]
        finally:
            reset_test_context(other_token)

        assert [c["test_id"] for c in chunks] == ["original-test"]
