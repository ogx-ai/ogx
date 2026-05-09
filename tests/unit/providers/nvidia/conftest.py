# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

mock_session = MagicMock()
mock_session.closed = False
mock_session.close = AsyncMock()
mock_session.__aenter__ = AsyncMock(return_value=mock_session)
mock_session.__aexit__ = AsyncMock()


@pytest.fixture(scope="session", autouse=True)
def patch_aiohttp_session():
    with patch("aiohttp.ClientSession", return_value=mock_session):
        yield


@pytest.fixture
def event_loop():
    """Create and provide a new event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def run_async():
    """Fixture to run async functions in tests."""

    def _run_async(coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    return _run_async
