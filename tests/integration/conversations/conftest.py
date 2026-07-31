# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pytest fixtures for conversation tests."""

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    ogx_client,
    openai_client,
    require_server,
)
