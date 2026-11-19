# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import httpx2
import pytest

from ogx.providers.utils.common.url_validation import PinnedIPTransport
from ogx.providers.utils.inference.prompt_adapter import localize_image_content


class TestLocalizeImageContentSSRF:
    async def test_rejects_loopback_url(self):
        with pytest.raises(ValueError, match="private IP"):
            await localize_image_content("http://127.0.0.1/image.png")

    async def test_rejects_metadata_url(self):
        with pytest.raises(ValueError, match="private IP"):
            await localize_image_content("http://169.254.169.254/latest/meta-data/")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    async def test_rejects_hostname_resolving_to_private(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 0)),
        ]
        with pytest.raises(ValueError, match="private IP"):
            await localize_image_content("http://rebind.example.com/image.png")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    async def test_fetches_public_url_through_pinned_transport(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]

        captured: dict = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                self.get = AsyncMock(
                    return_value=httpx2.Response(200, content=b"img", headers={"content-type": "image/png"})
                )
                return self

            async def __aexit__(self, *exc):
                return False

        with patch("ogx.providers.utils.inference.prompt_adapter.httpx2.AsyncClient", FakeClient):
            content, fmt = await localize_image_content("http://example.com/image.png")

        assert (content, fmt) == (b"img", "png")
        transport = captured["transport"]
        assert isinstance(transport, PinnedIPTransport)
        assert transport._pins == {"example.com": "93.184.216.34"}
