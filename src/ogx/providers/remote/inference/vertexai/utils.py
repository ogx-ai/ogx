# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any

import httpx
from google.genai import types as genai_types

from ogx.log import get_logger
from ogx.providers.utils.inference.http_client import _build_proxy_mounts, _build_ssl_context
from ogx.providers.utils.inference.model_registry import NetworkConfig, TimeoutConfig, TLSConfig

logger = get_logger(__name__, category="inference")


def resolve_timeout_ms(timeout: TimeoutConfig | float | None) -> int | None:
    """Convert a timeout config to milliseconds.

    Args:
        timeout: timeout as seconds float or TimeoutConfig object

    Returns:
        Timeout in milliseconds, or None if not configured
    """
    if timeout is None:
        return None

    if isinstance(timeout, TimeoutConfig):
        seconds = timeout.read if timeout.read is not None else timeout.connect
        if seconds is not None:
            return int(seconds * 1000)
        return None

    return int(timeout * 1000)


def resolve_ssl_verify(tls: TLSConfig) -> ssl.SSLContext | Path | str | bool:
    """Convert a TLSConfig into an ssl.SSLContext or verification path.

    Args:
        tls: TLS configuration object

    Returns:
        An SSLContext, certificate path, or boolean for verification
    """
    ssl_result = _build_ssl_context(tls)

    if ssl_result is False:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    if isinstance(ssl_result, Path):
        return str(ssl_result)

    return ssl_result


def build_httpx_kwargs(network_config: NetworkConfig) -> tuple[dict[str, Any], bool]:
    """Build httpx client keyword arguments from a NetworkConfig.

    Args:
        network_config: network configuration with TLS and proxy settings

    Returns:
        Tuple of (kwargs_dict, needs_httpx_client_bool)
    """
    httpx_kwargs: dict[str, Any] = {"follow_redirects": True}
    needs_httpx_client = False

    if network_config.tls:
        httpx_kwargs["verify"] = resolve_ssl_verify(network_config.tls)
        needs_httpx_client = True

    if network_config.proxy:
        if network_config.proxy.no_proxy:
            logger.warning("ProxyConfig.no_proxy is not supported by the VertexAI provider and will be ignored.")
        mounts = _build_proxy_mounts(network_config.proxy)
        if mounts:
            httpx_kwargs["mounts"] = mounts
            needs_httpx_client = True

    return httpx_kwargs, needs_httpx_client


def build_http_options(network_config: NetworkConfig | None) -> genai_types.HttpOptions | None:
    """Build Google GenAI HttpOptions from a NetworkConfig.

    Args:
        network_config: optional network configuration

    Returns:
        HttpOptions object or None if no options are configured
    """
    if network_config is None:
        return None

    kwargs: dict[str, Any] = {}

    if network_config.headers:
        kwargs["headers"] = network_config.headers

    if network_config.timeout is not None:
        timeout_ms = resolve_timeout_ms(network_config.timeout)
        if timeout_ms is not None:
            kwargs["timeout"] = timeout_ms

    httpx_kwargs, needs_httpx_client = build_httpx_kwargs(network_config)

    if needs_httpx_client:
        kwargs["httpx_async_client"] = httpx.AsyncClient(**httpx_kwargs)

    if not kwargs:
        return None

    return genai_types.HttpOptions(**kwargs)
