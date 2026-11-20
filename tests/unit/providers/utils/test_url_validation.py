# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import socket
from unittest.mock import AsyncMock, patch

import httpx2
import pytest

from ogx.providers.utils.common.url_validation import (
    PinnedIPTransport,
    check_url_scheme,
    pin_url,
)


class TestPinUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/image.png",
            "http://127.0.0.2:8080/path",
            "http://10.0.0.1/file",
            "http://172.16.0.1/file",
            "http://172.31.255.255/file",
            "http://192.168.1.1/file",
            "http://169.254.0.1/file",
            "http://[::1]/file",
            "http://[fc00::1]/file",
            "http://[fe80::1]/file",
            "http://[::ffff:127.0.0.1]/file",
            "http://0.0.0.0/file",
        ],
    )
    def test_rejects_private_ips(self, url):
        with pytest.raises(ValueError, match="private IP"):
            pin_url(url)

    def test_allows_public_ip(self):
        assert pin_url("http://8.8.8.8/file") == {"8.8.8.8": ["8.8.8.8"]}

    def test_allows_172_outside_private_range(self):
        assert pin_url("http://172.32.0.1/file") == {"172.32.0.1": ["172.32.0.1"]}

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_returns_all_validated_addresses_in_order(self, mock_getaddrinfo):
        # A dual-stack hostname keeps every validated address so a target
        # bound to a single family stays reachable (tried in order by the
        # transport).
        mock_getaddrinfo.return_value = [
            (10, 1, 6, "", ("::1", 0)),
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]
        assert pin_url("http://localhost/mcp", allow_private=True) == {"localhost": ["::1", "127.0.0.1"]}

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_rejects_hostname_resolving_to_private(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 0)),
        ]
        with pytest.raises(ValueError, match="private IP"):
            pin_url("http://evil.example.com/file")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_allows_hostname_resolving_to_public(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
        ]
        assert pin_url("http://example.com/file") == {"example.com": ["93.184.216.34"]}

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_rejects_hostname_with_mixed_public_and_private_results(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),
            (2, 1, 6, "", ("10.0.0.1", 0)),
        ]
        with pytest.raises(ValueError, match="private IP"):
            pin_url("http://mixed.example.com/file")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_allow_private_accepts_loopback_and_pins_it(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 0)),
        ]
        assert pin_url("http://mcp.internal/file", allow_private=True) == {"mcp.internal": ["127.0.0.1"]}

    def test_allow_private_accepts_metadata_ip_literal(self):
        assert pin_url("http://169.254.169.254/latest/meta-data/", allow_private=True) == {
            "169.254.169.254": ["169.254.169.254"]
        }

    def test_allow_private_still_rejects_unresolvable_hostnames(self):
        with pytest.raises(ValueError, match="Failed to resolve hostname"):
            pin_url("http://nonexistent.invalid/file", allow_private=True)

    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://8.8.8.8/file", "gopher://8.8.8.8/", "ws://example.com/mcp"],
    )
    def test_rejects_non_http_schemes(self, url):
        with pytest.raises(ValueError, match="only http and https"):
            pin_url(url)

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="Failed to parse hostname"):
            pin_url("http:///file")

    @patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
    def test_rejects_unresolvable_hostname(self, mock_getaddrinfo):
        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")
        with pytest.raises(ValueError, match="Failed to resolve hostname"):
            pin_url("http://nonexistent.invalid/file")


class TestCheckUrlScheme:
    def test_returns_hostname(self):
        assert check_url_scheme("https://example.com:8443/mcp") == "example.com"

    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://example.com/file", "ws://example.com/mcp"],
    )
    def test_rejects_non_http_schemes(self, url):
        with pytest.raises(ValueError, match="only http and https"):
            check_url_scheme(url)

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="Failed to parse hostname"):
            check_url_scheme("http:///file")


class TestPinnedIPTransport:
    """The transport must route to the pinned IP while the TLS handshake
    still targets the original hostname (SNI + certificate verification).
    httpcore reads the SNI name from the ``sni_hostname`` request extension,
    falling back to the (already-swapped) URL host without it."""

    async def _drive(self, url: str, pins: dict[str, str], **request_kwargs: object) -> httpx2.Request:
        transport = PinnedIPTransport(pins)
        request = httpx2.Request("GET", url, **request_kwargs)  # type: ignore[arg-type]
        with patch.object(httpx2.AsyncHTTPTransport, "handle_async_request", new_callable=AsyncMock) as mock_super:
            mock_super.return_value = httpx2.Response(200)
            await transport.handle_async_request(request)
        assert mock_super.call_count == 1
        return mock_super.call_args.args[0]

    async def test_https_request_pins_ip_and_keeps_sni(self):
        sent = await self._drive(
            "https://mcp.example.com:8443/mcp",
            {"mcp.example.com": ["93.184.216.34"]},
            headers={"host": "mcp.example.com:8443"},
        )
        assert sent.url.host == "93.184.216.34"
        assert sent.url.port == 8443
        assert sent.extensions.get("sni_hostname") == "mcp.example.com"
        assert sent.headers["host"] == "mcp.example.com:8443"

    async def test_http_request_pins_ip_without_sni(self):
        sent = await self._drive("http://example.com/image.png", {"example.com": ["93.184.216.34"]})
        assert sent.url.host == "93.184.216.34"
        assert sent.extensions.get("sni_hostname") is None
        # No client-set Host header: the fallback keeps the original hostname.
        assert sent.headers["host"] == "example.com"

    async def test_falls_back_to_next_validated_address(self):
        """Models the dual-stack localhost case: a hostname pinned to
        ["::1", "127.0.0.1"] (or the 127.0.0.2/127.0.0.1 analogue below)
        must stay reachable when the target only listens on one family."""
        import asyncio

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.read(1024)
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        # Only 127.0.0.1 listens: the pinned 127.0.0.2 is tried first and
        # refused, then the fallback succeeds.
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            transport = PinnedIPTransport({"127.0.0.1": ["127.0.0.2", "127.0.0.1"]})
            async with httpx2.AsyncClient(transport=transport, timeout=5.0) as client:
                response = await client.get(f"http://127.0.0.1:{port}/")
            assert response.status_code == 200
            assert response.text == "ok"
        finally:
            server.close()
            await server.wait_closed()
