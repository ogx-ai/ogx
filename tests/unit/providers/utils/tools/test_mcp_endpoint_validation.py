# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from ogx.providers.utils.common.url_validation import resolve_public_ips
from ogx.providers.utils.tools.mcp import (
    _pinned_sse_client_factory,
    _PinnedIPTransport,
    list_mcp_tools,
    validate_mcp_endpoint,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1/mcp",
        "http://10.0.0.1/mcp",
        "http://192.168.1.1/mcp",
        "http://172.16.0.1/mcp",
        "http://0.0.0.0/mcp",
        "http://100.64.0.1/mcp",  # CGNAT
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/mcp",
        "http://[fe80::1]/mcp",  # IPv6 link-local
        "http://[fd00::1]/mcp",  # IPv6 unique-local
        "http://[::ffff:127.0.0.1]/mcp",  # IPv4-mapped loopback
        "file:///etc/passwd",
        "gopher://example.com/1",
        "ftp://example.com/mcp",
    ],
)
def test_validate_mcp_endpoint_rejects_private_and_non_http_urls(endpoint: str) -> None:
    with pytest.raises(ValueError, match="Failed to"):
        validate_mcp_endpoint(endpoint)


def _addrinfo_results(ips: list[str]) -> list[tuple]:
    entries = []
    for ip in ips:
        if ":" in ip:
            entries.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0)))
        else:
            entries.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)))
    return entries


def test_resolve_public_ips_ip_literal_skips_dns() -> None:
    assert resolve_public_ips("8.8.8.8") == ["8.8.8.8"]


@patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
def test_resolve_public_ips_returns_all_public_answers_deduplicated(mock_getaddrinfo: MagicMock) -> None:
    mock_getaddrinfo.return_value = _addrinfo_results(["93.184.216.34", "93.184.216.34", "2606:2800:220:1::244"])
    assert resolve_public_ips("mcp.example.com") == ["93.184.216.34", "2606:2800:220:1::244"]
    mock_getaddrinfo.assert_called_once()


@patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
def test_resolve_public_ips_rejects_when_any_answer_is_private(mock_getaddrinfo: MagicMock) -> None:
    mock_getaddrinfo.return_value = _addrinfo_results(["93.184.216.34", "127.0.0.1"])
    with pytest.raises(ValueError, match="private IP addresses are not allowed"):
        resolve_public_ips("rebind.example.com")


@patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo", side_effect=socket.gaierror("unresolvable"))
def test_resolve_public_ips_unresolvable_hostname(mock_getaddrinfo: MagicMock) -> None:
    with pytest.raises(ValueError, match="Failed to resolve hostname"):
        resolve_public_ips("no-such-host.example.com")


@patch("ogx.providers.utils.common.url_validation.socket.getaddrinfo")
def test_pre_connection_recheck_catches_dns_rebinding(mock_getaddrinfo: MagicMock) -> None:
    """The pre-connect re-resolution must catch a rebinding host that served a public
    answer during the initial validation but a private one when connecting."""
    mock_getaddrinfo.side_effect = [
        _addrinfo_results(["93.184.216.34"]),  # resolution during validation: public
        _addrinfo_results(["127.0.0.1"]),  # re-resolution right before connecting: private
    ]
    validate_mcp_endpoint("http://rebind.example.com/sse")  # passes
    with pytest.raises(ValueError, match="private IP addresses are not allowed"):
        validate_mcp_endpoint("http://rebind.example.com/sse")  # fresh resolution catches the rebind
    assert mock_getaddrinfo.call_count == 2


@patch("ogx.providers.utils.tools.mcp.resolve_public_ips", return_value=["93.184.216.34"])
def test_validate_mcp_endpoint_returns_hostname_to_ip_map(mock_resolve: MagicMock) -> None:
    assert validate_mcp_endpoint("http://mcp.example.com/sse") == {"mcp.example.com": "93.184.216.34"}
    assert validate_mcp_endpoint("https://mcp.example.com/mcp") == {"mcp.example.com": "93.184.216.34"}
    assert mock_resolve.call_count == 2


@patch(
    "ogx.providers.utils.tools.mcp.resolve_public_ips",
    side_effect=ValueError(
        "Failed to fetch URL: requests to private IP addresses are not allowed (rebind.example.com)"
    ),
)
def test_validate_mcp_endpoint_rejects_private_rebind(mock_resolve: MagicMock) -> None:
    with pytest.raises(ValueError, match="private IP addresses are not allowed"):
        validate_mcp_endpoint("http://rebind.example.com/sse")


async def test_pinned_ip_transport_connects_to_pinned_address_and_keeps_host_header() -> None:
    received: dict[str, str] = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        received["head"] = head.decode()
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        # Nothing listens on 127.0.0.2: the request can only succeed if the
        # transport reroutes the connection to the pinned address.
        transport = _PinnedIPTransport({"127.0.0.2": "127.0.0.1"})
        async with httpx2.AsyncClient(transport=transport, timeout=5.0) as client:
            response = await client.get(f"http://127.0.0.2:{port}/sse")
        assert response.status_code == 200
        assert response.text == "ok"
        assert f"host: 127.0.0.2:{port}" in received["head"].lower()
    finally:
        server.close()
        await server.wait_closed()


async def test_pinned_ip_transport_rejects_non_pinned_hosts() -> None:
    """A pinned transport must fail closed: a host without a pin (e.g. from a
    server-supplied SSE endpoint event) is rejected instead of being resolved."""
    transport = _PinnedIPTransport({"127.0.0.2": "127.0.0.1"})
    async with httpx2.AsyncClient(transport=transport, timeout=5.0) as client:
        with pytest.raises(ValueError, match="non-pinned hosts are not allowed"):
            await client.get("http://127.0.0.1:9/sse")


async def test_pinned_sse_client_factory_uses_pinned_transport() -> None:
    factory = _pinned_sse_client_factory({"127.0.0.2": "127.0.0.1"})
    client = factory(headers={"Authorization": "Bearer token"}, timeout=httpx2.Timeout(5.0))
    try:
        assert isinstance(client, httpx2.AsyncClient)
        assert isinstance(client._transport, _PinnedIPTransport)
    finally:
        await client.aclose()


@patch("ogx.providers.utils.tools.mcp.resolve_public_ips", return_value=["93.184.216.34"])
def test_validate_mcp_endpoint_allows_public_https_url(mock_resolve: MagicMock) -> None:
    assert validate_mcp_endpoint("https://mcp.example.com/mcp") == {"mcp.example.com": "93.184.216.34"}
    mock_resolve.assert_called_once_with("mcp.example.com")


async def test_list_mcp_tools_still_allows_private_admin_endpoints() -> None:
    """Admin-configured connectors/toolgroups may use private MCP URLs.

    SSRF validation for caller-supplied Responses server_url lives in the
    Responses path, not in the shared MCP helpers.
    """
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = MagicMock(tools=[])
    mock_wrapper = MagicMock()
    mock_wrapper.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_wrapper.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("ogx.providers.utils.tools.mcp.client_wrapper", mock_wrapper):
        result = await list_mcp_tools(endpoint="http://10.0.0.5/mcp")

    assert result.data == []
    mock_wrapper.assert_called_once()
