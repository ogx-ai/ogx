# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core.library_client import OGXAsLibraryClient

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import mock_sentinel  # noqa: F401


@pytest.fixture(scope="session")
def connect_base_url(ogx_client):
    """The OGX server URL for `ogx connect --url`, skipping library client mode."""
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("Connect CLI smoke tests require server mode")
    return f"{ogx_client.base_url}/v1"
