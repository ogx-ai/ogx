# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from ogx.core.library_client import OGXAsLibraryClient

# The web search backends of the ci-tests web-search.yaml config, in config
# order (which is the routing table's resolution order).
_SEARCH_BACKEND_ENV_VARS = (
    ("BRAVE_SEARCH_API_KEY", "brave-search"),
    ("BING_API_KEY", "bing-search"),
    ("EXA_API_KEY", "exa-search"),
    ("NIMBLE_API_KEY", "nimble-search"),
    ("SERPLY_API_KEY", "serply-search"),
    ("SEARCHAPI_API_KEY", "searchapi-search"),
    ("TAVILY_SEARCH_API_KEY", "tavily-search"),
)


def active_search_provider() -> str:
    """The web search backend the server resolves to for this run.

    Mirrors the routing table's resolution: the first configured backend in
    config order (a backend is configured when its API key env var is set),
    else the placeholder.
    """
    for env_var, provider_id in _SEARCH_BACKEND_ENV_VARS:
        if os.environ.get(env_var):
            return provider_id
    return "placeholder"


def pytest_configure(config):
    """Disable stderr pipe to prevent Rich logging from blocking on buffer saturation.

    When tests intentionally trigger errors, the server writes extensive error logs to stderr.
    Since stderr is piped to a subprocess buffer with limited pre-allocated memory, the buffer
    can fill up and cause the server's logging calls to block, making the server unresponsive.
    Most tests don't need to inspect server logs, so disabling the pipe prevents this deadlock.

    This runs before any fixtures, ensuring the server starts with stderr disabled.
    """
    os.environ["OGX_TEST_LOG_STDERR"] = "0"


@pytest.fixture(autouse=True)
def _track_test_context(request):
    """Scope this suite's recordings to the active web search backend.

    Overrides the integration-wide fixture of the same name. The model samples a
    different web search query on each live run, and the post-search inference
    requests embed the (backend-specific) search results, so the recordings are
    only self-consistent within one backend's lane. Suffixing the test context
    keeps every lane's record/replay chain isolated.
    """
    from ogx.core.testing_context import reset_test_context, set_test_context

    token = set_test_context(f"{request.node.nodeid}::{active_search_provider()}")

    yield

    reset_test_context(token)


@pytest.fixture
def responses_client(compat_client):
    """Provide a client for web search tests, skipping library client mode."""
    if isinstance(compat_client, OGXAsLibraryClient):
        pytest.skip("Web search tests are not supported in library client mode")
    return compat_client
