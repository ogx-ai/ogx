# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for test-ID injection: the OgxClient._prepare_request patch, and the
httpx2 event-hook clients used for openai.OpenAI/AsyncOpenAI and langchain (#6627).

Injection is unconditional -- it applies in every stack mode (#6667) -- so these tests
only distinguish an active test context (test_id is set) from no test context at all."""

import json

import httpx2
import pytest
from ogx_client import OgxClient
from openai import OpenAI

from ogx.core.testing_context import reset_test_context, set_test_context
from ogx.testing import api_recorder
from ogx.testing.api_recorder import (
    build_test_id_async_http_client,
    build_test_id_http_client,
    patch_httpx_for_test_id,
)

PROVIDER_DATA_HEADER = "X-OGX-Provider-Data"
TEST_ID = "tests/unit/testing/test_api_recorder.py::test"


@pytest.fixture
def unpatched_ogx_client():
    """Undo the process-wide OgxClient._prepare_request patch that a test installs."""
    ogx_prepare_request = OgxClient._prepare_request
    api_recorder._prepare_request_originals.clear()
    yield
    OgxClient._prepare_request = ogx_prepare_request
    api_recorder._prepare_request_originals.clear()


@pytest.fixture
def test_context():
    token = set_test_context(TEST_ID)
    yield
    reset_test_context(token)


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "http://localhost:8321/v1/responses", json={})


def _ogx_client() -> OgxClient:
    return OgxClient(base_url="http://localhost:8321")


class TestPatchOgxClient:
    """OgxClient._prepare_request is our own generated client's documented request hook,
    not a private third-party SDK internal -- patching it is unaffected by #6627."""

    def test_original_still_runs(self, unpatched_ogx_client):
        """Patching must not swallow whatever OgxClient's own _prepare_request does."""
        prepared = []
        OgxClient._prepare_request = lambda self, request: prepared.append(request)

        patch_httpx_for_test_id()

        request = _request()
        _ogx_client()._prepare_request(request)

        assert prepared == [request]

    def test_test_id_injected_when_a_test_context_is_active(self, unpatched_ogx_client, test_context):
        patch_httpx_for_test_id()

        request = _request()
        _ogx_client()._prepare_request(request)

        assert json.loads(request.headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}

    def test_existing_provider_data_is_preserved(self, unpatched_ogx_client, test_context):
        patch_httpx_for_test_id()

        request = _request()
        request.headers[PROVIDER_DATA_HEADER] = json.dumps({"api_key": "abc"})
        _ogx_client()._prepare_request(request)

        assert json.loads(request.headers[PROVIDER_DATA_HEADER]) == {"api_key": "abc", "__test_id": TEST_ID}

    def test_no_injection_without_an_active_test_context(self, unpatched_ogx_client):
        patch_httpx_for_test_id()

        request = _request()
        _ogx_client()._prepare_request(request)

        assert PROVIDER_DATA_HEADER not in request.headers

    def test_malformed_existing_provider_data_is_discarded_rather_than_raising(
        self, unpatched_ogx_client, test_context
    ):
        """Injection delegates its header parsing to stamp_test_id_into_headers() (shared with
        the in-process library-client path), which must not raise on a header it can't parse."""
        patch_httpx_for_test_id()

        request = _request()
        request.headers[PROVIDER_DATA_HEADER] = "not-valid-json"
        _ogx_client()._prepare_request(request)

        assert json.loads(request.headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}

    def test_originals_survive_api_recording_clearing_original_methods(self, unpatched_ogx_client):
        """api_recording() reassigns and clears _original_methods, so the patch must not read it."""
        calls = []
        OgxClient._prepare_request = lambda self, request: calls.append(request)

        patch_httpx_for_test_id()
        api_recorder._original_methods.clear()

        request = _request()
        _ogx_client()._prepare_request(request)

        assert calls == [request]

    def test_patching_twice_does_not_stack_wrappers(self, unpatched_ogx_client):
        patch_httpx_for_test_id()
        patched = OgxClient._prepare_request

        patch_httpx_for_test_id()

        assert OgxClient._prepare_request is patched


class TestBuildTestIdHttpClient:
    """build_test_id_http_client()/build_test_id_async_http_client() use httpx2's own public,
    documented event_hooks mechanism -- no openai/langchain internals involved -- so the same
    two functions cover openai.OpenAI, openai.AsyncOpenAI, and langchain's ChatOpenAI uniformly."""

    def test_returns_configured_httpx_client(self):
        client = build_test_id_http_client()
        assert isinstance(client, httpx2.Client)
        client.close()

    async def test_returns_configured_httpx_async_client(self):
        client = build_test_id_async_http_client()
        assert isinstance(client, httpx2.AsyncClient)
        await client.aclose()

    def test_sync_client_injects_test_id(self, test_context):
        captured: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return httpx2.Response(200, json={})

        with build_test_id_http_client(transport=httpx2.MockTransport(handler)) as client:
            client.post("http://localhost:8321/v1/responses", json={})

        assert json.loads(captured[0].headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}

    async def test_async_client_injects_test_id(self, test_context):
        captured: list[httpx2.Request] = []

        async def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return httpx2.Response(200, json={})

        async with build_test_id_async_http_client(transport=httpx2.MockTransport(handler)) as client:
            await client.post("http://localhost:8321/v1/responses", json={})

        assert json.loads(captured[0].headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}

    def test_no_injection_without_an_active_test_context(self):
        captured: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return httpx2.Response(200, json={})

        with build_test_id_http_client(transport=httpx2.MockTransport(handler)) as client:
            client.post("http://localhost:8321/v1/responses", json={})

        assert PROVIDER_DATA_HEADER not in captured[0].headers

    def test_preserves_callers_own_event_hooks(self, test_context):
        """Passing event_hooks= must add our hook alongside the caller's, not replace it."""
        own_hook_calls: list[httpx2.Request] = []

        def own_hook(request: httpx2.Request) -> None:
            own_hook_calls.append(request)

        captured: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return httpx2.Response(200, json={})

        with build_test_id_http_client(
            event_hooks={"request": [own_hook]}, transport=httpx2.MockTransport(handler)
        ) as client:
            client.post("http://localhost:8321/v1/responses", json={})

        assert len(own_hook_calls) == 1
        assert json.loads(captured[0].headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}

    def test_openai_client_with_http_client_gets_test_id(self, test_context):
        """End-to-end: a real openai.OpenAI() constructed with http_client=build_test_id_http_client()
        stamps __test_id, without touching OpenAI._prepare_request at all."""
        captured: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(request)
            return httpx2.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                    ],
                },
            )

        client = OpenAI(
            api_key="fake",
            base_url="http://localhost:8321/v1",
            http_client=build_test_id_http_client(transport=httpx2.MockTransport(handler)),
            max_retries=0,
        )

        client.chat.completions.create(model="test", messages=[{"role": "user", "content": "hi"}])

        assert json.loads(captured[0].headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}
