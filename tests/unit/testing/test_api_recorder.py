# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the _prepare_request patches that carry the test ID in server mode."""

import json

import httpx
import pytest
from ogx_client import OgxClient
from openai import OpenAI

from ogx.core.testing_context import reset_test_context, set_test_context
from ogx.testing import api_recorder
from ogx.testing.api_recorder import patch_httpx_for_test_id

PROVIDER_DATA_HEADER = "X-OGX-Provider-Data"
TEST_ID = "tests/unit/testing/test_api_recorder.py::test"


@pytest.fixture
def unpatched_clients():
    """Undo the process-wide _prepare_request patches that a test installs."""
    ogx_prepare_request = OgxClient._prepare_request
    openai_prepare_request = OpenAI._prepare_request
    api_recorder._prepare_request_originals.clear()
    yield
    OgxClient._prepare_request = ogx_prepare_request
    OpenAI._prepare_request = openai_prepare_request
    api_recorder._prepare_request_originals.clear()


@pytest.fixture
def test_context():
    token = set_test_context(TEST_ID)
    yield
    reset_test_context(token)


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://localhost:8321/v1/responses", json={})


def _ogx_client() -> OgxClient:
    return OgxClient(base_url="http://localhost:8321")


def _openai_client() -> OpenAI:
    return OpenAI(api_key="not-a-real-key", base_url="http://localhost:8321/v1")


def test_each_client_calls_only_its_own_original(unpatched_clients):
    """The patch must not pass an OgxClient to OpenAI._prepare_request, or vice versa."""
    calls = []

    def ogx_original(self, request):
        calls.append(("ogx", self))

    def openai_original(self, request):
        calls.append(("openai", self))

    OgxClient._prepare_request = ogx_original
    OpenAI._prepare_request = openai_original

    patch_httpx_for_test_id()

    ogx_client = _ogx_client()
    openai_client = _openai_client()
    ogx_client._prepare_request(_request())
    openai_client._prepare_request(_request())

    assert calls == [("ogx", ogx_client), ("openai", openai_client)]


def test_ogx_client_does_not_touch_openai_instance_state(unpatched_clients):
    """openai >= 2.44.0 dereferences self._provider_runtime, which OgxClient does not have."""
    patch_httpx_for_test_id()

    assert _ogx_client()._prepare_request(_request()) is None


def test_openai_original_still_runs(unpatched_clients):
    """Patching must not swallow whatever openai's own _prepare_request does."""
    prepared = []
    OpenAI._prepare_request = lambda self, request: prepared.append(request)

    patch_httpx_for_test_id()

    request = _request()
    _openai_client()._prepare_request(request)

    assert prepared == [request]


@pytest.mark.parametrize("client_factory", [_ogx_client, _openai_client], ids=["ogx", "openai"])
def test_test_id_injected_in_server_mode(unpatched_clients, test_context, monkeypatch, client_factory):
    monkeypatch.setenv("OGX_TEST_STACK_CONFIG_TYPE", "server")
    patch_httpx_for_test_id()

    request = _request()
    client_factory()._prepare_request(request)

    assert json.loads(request.headers[PROVIDER_DATA_HEADER]) == {"__test_id": TEST_ID}


def test_existing_provider_data_is_preserved(unpatched_clients, test_context, monkeypatch):
    monkeypatch.setenv("OGX_TEST_STACK_CONFIG_TYPE", "server")
    patch_httpx_for_test_id()

    request = _request()
    request.headers[PROVIDER_DATA_HEADER] = json.dumps({"api_key": "abc"})
    _ogx_client()._prepare_request(request)

    assert json.loads(request.headers[PROVIDER_DATA_HEADER]) == {"api_key": "abc", "__test_id": TEST_ID}


def test_no_injection_in_library_client_mode(unpatched_clients, test_context, monkeypatch):
    monkeypatch.setenv("OGX_TEST_STACK_CONFIG_TYPE", "library_client")
    patch_httpx_for_test_id()

    request = _request()
    _ogx_client()._prepare_request(request)

    assert PROVIDER_DATA_HEADER not in request.headers


def test_originals_survive_api_recording_clearing_original_methods(unpatched_clients):
    """api_recording() reassigns and clears _original_methods, so the patch must not read it."""
    calls = []
    OgxClient._prepare_request = lambda self, request: calls.append(request)

    patch_httpx_for_test_id()
    api_recorder._original_methods.clear()

    request = _request()
    _ogx_client()._prepare_request(request)

    assert calls == [request]


def test_patching_twice_does_not_stack_wrappers(unpatched_clients):
    patch_httpx_for_test_id()
    patched = OgxClient._prepare_request

    patch_httpx_for_test_id()

    assert OgxClient._prepare_request is patched
