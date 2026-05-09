# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ipaddress
import socket
from urllib.parse import urlparse

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _is_non_public_ip(addr: _IPAddress) -> bool:
    """Return True when an address is not publicly routable."""
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return _is_non_public_ip(addr.ipv4_mapped)
    return not addr.is_global


def _raise_blocked_ip_error(hostname: str) -> None:
    raise ValueError(f"Failed to fetch URL: requests to private IP addresses are not allowed ({hostname})")


def validate_url_not_private(url: str) -> None:
    """Reject URLs that resolve to private or loopback IP addresses.

    Raises:
        ValueError: If the URL hostname resolves to a blocked IP range or cannot be resolved.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Failed to parse hostname from URL: {url}")

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"Failed to resolve hostname: {hostname}") from exc
        for info in infos:
            addr = ipaddress.ip_address(info[4][0])
            if _is_non_public_ip(addr):
                _raise_blocked_ip_error(hostname)
        return

    if _is_non_public_ip(addr):
        _raise_blocked_ip_error(hostname)
