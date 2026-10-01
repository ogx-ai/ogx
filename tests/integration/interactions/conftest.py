# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import pytest
from google import genai
from google.genai import types

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
def interactions_base_url(ogx_client):
    """Provide the base URL for the Interactions API, skipping library client mode."""
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("Interactions API tests are not supported in library client mode")
    return ogx_client.base_url


@pytest.fixture
def genai_client(interactions_base_url):
    """Provide a Google GenAI client configured to point at the OGX server.

    interactions_base_url above already skips outside server mode, so this always runs
    against a real server and the test ID always needs to cross via the header.
    """
    headers = {}
    test_id = get_test_context()
    if test_id:
        headers["X-OGX-Provider-Data"] = json.dumps({"__test_id": test_id})

    client = genai.Client(
        api_key="no-key-required",
        http_options=types.HttpOptions(
            base_url=str(interactions_base_url),
            api_version="v1alpha",
            headers=headers,
        ),
    )
    return client
