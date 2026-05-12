# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import pytest
from anthropic import Anthropic

from ogx.core.library_client import OGXAsLibraryClient
from ogx.core.testing_context import get_test_context

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    openai_client,
    require_server,
)


def pytest_configure(config):
    """Disable stderr pipe to prevent Rich logging from blocking on buffer saturation."""
    os.environ["OGX_TEST_LOG_STDERR"] = "0"


@pytest.fixture(scope="session")
def models_base_url(ogx_client):
    """Provide the base URL for the Models API, skipping library client mode."""
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("Models SDK tests are not supported in library client mode")
    return str(ogx_client.base_url)


def _sdk_provider_data_headers() -> dict[str, str]:
    """Inject test ID for server-mode recording isolation when available."""
    test_id = get_test_context()
    if not test_id:
        return {}
    provider_data = {"__test_id": test_id}
    return {"X-OGX-Provider-Data": json.dumps(provider_data)}


@pytest.fixture
def anthropic_client(models_base_url):
    """Provide an Anthropic SDK client configured to point at the OGX server."""
    client = Anthropic(
        api_key="fake",
        base_url=models_base_url,
        default_headers=_sdk_provider_data_headers(),
        max_retries=0,
        timeout=30.0,
    )
    yield client
    client.close()


@pytest.fixture
def google_genai_client(models_base_url):
    """Provide a Google GenAI SDK client configured to point at the OGX server."""
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")
    client = genai.Client(
        api_key="no-key-required",
        http_options=types.HttpOptions(
            base_url=models_base_url,
            api_version="v1",
            headers=_sdk_provider_data_headers(),
            # google-genai HttpOptions timeout is in milliseconds.
            timeout=30_000,
        ),
    )
    yield client
    client.close()
