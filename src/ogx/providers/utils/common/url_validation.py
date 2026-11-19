# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import httpx2

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _is_non_public_ip(addr: _IPAddress) -> bool:
    """Return True when an address is not publicly routable."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return _is_non_public_ip(addr.ipv4_mapped)
    return not addr.is_global


def _resolve_ips(hostname: str) -> list[str]:
    """Resolve a hostname (or IP literal) to the list of addresses it maps to."""
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"Failed to resolve hostname: {hostname}") from exc
        return [str(info[4][0]) for info in infos]
    return [str(addr)]


def resolve_public_ips(hostname: str) -> list[str]:
    """Resolve a hostname and return the public IP addresses it maps to.

    Raises:
        ValueError: If the hostname is unresolvable or any resolved address is
            not publicly routable (private, loopback, link-local, reserved, ...).
    """
    ips = _resolve_ips(hostname)
    for ip in ips:
        if _is_non_public_ip(ipaddress.ip_address(ip)):
            raise ValueError(f"Failed to fetch URL: requests to private IP addresses are not allowed ({hostname})")
    return list(dict.fromkeys(ips))


def check_url_scheme(url: str) -> str:
    """Validate that url is an http(s) URL and return its hostname.

    Raises:
        ValueError: If the scheme is not http/https or the URL has no hostname.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Failed to fetch URL: only http and https URLs are allowed ({url})")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Failed to parse hostname from URL: {url}")
    return hostname


def pin_url(url: str, allow_private: bool = False) -> dict[str, str]:
    """Validate a URL and pin its hostname to a resolved IP address.

    Returns a ``{hostname: ip}`` map for use with :class:`PinnedIPTransport`,
    which routes the connection to the validated address so a DNS-rebinding
    swap between this check and the TCP connect cannot redirect the request to
    a different address. Callers must re-invoke this (with a fresh resolution)
    right before each connection.

    By default only http(s) URLs whose host resolves exclusively to publicly
    routable addresses are accepted. ``allow_private=True`` keeps the scheme
    and resolution checks but permits non-public addresses, for destinations
    that are trusted by configuration rather than supplied by a caller.

    Raises:
        ValueError: If the scheme is not http/https, the hostname is
            unresolvable, or (unless ``allow_private``) any resolved address
            is not publicly routable.
    """
    hostname = check_url_scheme(url)
    if allow_private:
        ips = _resolve_ips(hostname)
    else:
        ips = resolve_public_ips(hostname)
    return {hostname: ips[0]}


class PinnedIPTransport(httpx2.AsyncHTTPTransport):
    """Route requests for pinned hosts to a validated IP address.

    The Host header (and, for https, the TLS SNI name) is kept at the original
    hostname so the target server sees the request as unremarkable.

    Fails closed: any request to a host without a pin is rejected rather than
    resolved, so a server-supplied URL (e.g. an SSE endpoint event) cannot
    steer the connection to an address that never passed validation.
    """

    def __init__(self, pins: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pins = pins

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        host = request.url.host
        pinned = self._pins.get(host)
        if pinned is None:
            raise ValueError(f"Failed to connect: requests to non-pinned hosts are not allowed ({host})")
        # Keep the original Host header (includes a non-default port) so the
        # target server sees the request as unremarkable.
        host_header = request.headers.get("host", host)
        request.url = request.url.copy_with(host=pinned)
        request.headers["host"] = host_header
        if request.url.scheme == "https":
            request.extensions["sni_hostname"] = host
        return await super().handle_async_request(request)
