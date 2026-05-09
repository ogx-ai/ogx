# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import json
import ssl
from pathlib import Path
from typing import Any

import httpx
from openai._base_client import DefaultAsyncHttpxClient

from ogx.log import get_logger
from ogx.providers.utils.inference.model_registry import (
    NetworkConfig,
    ProxyConfig,
    TimeoutConfig,
    TLSConfig,
)

logger = get_logger(name=__name__, category="providers::utils")


_NETWORK_CONFIG_FINGERPRINT_ATTR = "_ogx_network_config_fingerprint"


def network_config_fingerprint(network_config: NetworkConfig) -> str:
    dumped = json.dumps(network_config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _get_client_network_fingerprint(existing_client: httpx.AsyncClient | DefaultAsyncHttpxClient) -> str | None:
    if isinstance(existing_client, DefaultAsyncHttpxClient):
        underlying_client = existing_client._client  # type: ignore[union-attr,attr-defined]
        return getattr(underlying_client, _NETWORK_CONFIG_FINGERPRINT_ATTR, None)
    return getattr(existing_client, _NETWORK_CONFIG_FINGERPRINT_ATTR, None)


def set_client_network_fingerprint(client: httpx.AsyncClient | DefaultAsyncHttpxClient, fingerprint: str) -> None:
    if isinstance(client, DefaultAsyncHttpxClient):
        underlying_client = client._client  # type: ignore[union-attr,attr-defined]
        setattr(underlying_client, _NETWORK_CONFIG_FINGERPRINT_ATTR, fingerprint)
    else:
        setattr(client, _NETWORK_CONFIG_FINGERPRINT_ATTR, fingerprint)


def _build_ssl_context(tls_config: TLSConfig) -> ssl.SSLContext | bool | str:
    """
    Build an SSL context from TLS configuration.

    Returns:
        - ssl.SSLContext if advanced options (min_version, ciphers, or mTLS) are configured
        - str if only a CA bundle path is specified (httpx requires str, not Path)
        - bool if only verify is specified as boolean
    """
    has_advanced_options = (
        tls_config.min_version is not None or tls_config.ciphers is not None or tls_config.client_cert is not None
    )

    if not has_advanced_options:
        if isinstance(tls_config.verify, Path):
            return str(tls_config.verify)
        return tls_config.verify

    ctx = ssl.create_default_context()

    if isinstance(tls_config.verify, Path):
        ctx.load_verify_locations(str(tls_config.verify))
    elif not tls_config.verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    if tls_config.min_version:
        if tls_config.min_version == "TLSv1.2":
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        elif tls_config.min_version == "TLSv1.3":
            ctx.minimum_version = ssl.TLSVersion.TLSv1_3

    if tls_config.ciphers:
        ctx.set_ciphers(":".join(tls_config.ciphers))

    if tls_config.client_cert and tls_config.client_key:
        ctx.load_cert_chain(certfile=str(tls_config.client_cert), keyfile=str(tls_config.client_key))

    return ctx


def _build_proxy_mounts(proxy_config: ProxyConfig) -> dict[str, httpx.AsyncHTTPTransport] | None:
    """
    Build httpx proxy mounts from proxy configuration.

    Returns:
        Dictionary of proxy mounts for httpx, or None if no proxies configured
    """
    transport_kwargs: dict[str, Any] = {}
    if proxy_config.cacert:
        # Convert Path to string for httpx
        transport_kwargs["verify"] = str(proxy_config.cacert)

    if proxy_config.url:
        # Convert HttpUrl to string for httpx
        proxy_url = str(proxy_config.url)
        return {
            "http://": httpx.AsyncHTTPTransport(proxy=proxy_url, **transport_kwargs),
            "https://": httpx.AsyncHTTPTransport(proxy=proxy_url, **transport_kwargs),
        }

    mounts = {}
    if proxy_config.http:
        mounts["http://"] = httpx.AsyncHTTPTransport(proxy=str(proxy_config.http), **transport_kwargs)
    if proxy_config.https:
        mounts["https://"] = httpx.AsyncHTTPTransport(proxy=str(proxy_config.https), **transport_kwargs)

    return mounts if mounts else None


def build_network_client_kwargs(network_config: NetworkConfig | None) -> dict[str, Any]:
    """
    Build httpx.AsyncClient kwargs from network configuration.

    This function creates the appropriate kwargs to pass to httpx.AsyncClient
    based on the provided NetworkConfig, without creating the client itself.

    Args:
        network_config: Network configuration including TLS, proxy, and timeout settings

    Returns:
        Dictionary of kwargs to pass to httpx.AsyncClient constructor
    """
    if network_config is None:
        return {}

    client_kwargs: dict[str, Any] = {}

    if network_config.tls:
        ssl_context = _build_ssl_context(network_config.tls)
        client_kwargs["verify"] = ssl_context

    if network_config.proxy:
        mounts = _build_proxy_mounts(network_config.proxy)
        if mounts:
            client_kwargs["mounts"] = mounts

    if network_config.timeout is not None:
        if isinstance(network_config.timeout, TimeoutConfig):
            # httpx.Timeout requires all four parameters (connect, read, write, pool)
            # to be set explicitly, or a default timeout value
            timeout_kwargs: dict[str, float | None] = {
                "connect": network_config.timeout.connect,
                "read": network_config.timeout.read,
                "write": None,
                "pool": None,
            }
            client_kwargs["timeout"] = httpx.Timeout(**timeout_kwargs)
        else:
            client_kwargs["timeout"] = httpx.Timeout(network_config.timeout)

    if network_config.headers:
        client_kwargs["headers"] = network_config.headers

    return client_kwargs


def _extract_client_config(existing_client: httpx.AsyncClient | DefaultAsyncHttpxClient) -> dict[str, Any]:
    """
    Extract configuration (auth, headers) from an existing http_client.

    Args:
        existing_client: Existing httpx client (may be DefaultAsyncHttpxClient)

    Returns:
        Dictionary with extracted auth and headers, if available
    """
    config: dict[str, Any] = {}

    # Extract from DefaultAsyncHttpxClient
    if isinstance(existing_client, DefaultAsyncHttpxClient):
        underlying_client = existing_client._client  # type: ignore[union-attr,attr-defined]
        if hasattr(underlying_client, "_auth"):
            config["auth"] = underlying_client._auth  # type: ignore[attr-defined]
        if hasattr(existing_client, "_headers"):
            config["headers"] = existing_client._headers  # type: ignore[attr-defined]
    else:
        # Extract from plain httpx.AsyncClient
        if hasattr(existing_client, "_auth"):
            config["auth"] = existing_client._auth  # type: ignore[attr-defined]
        if hasattr(existing_client, "_headers"):
            config["headers"] = existing_client._headers  # type: ignore[attr-defined]

    return config


def _merge_network_config_into_client(
    existing_client: httpx.AsyncClient | DefaultAsyncHttpxClient, network_config: NetworkConfig | None
) -> httpx.AsyncClient | DefaultAsyncHttpxClient:
    """
    Merge network configuration into an existing http_client.

    Extracts auth and headers from the existing client, merges with network config,
    and creates a new client with all settings combined.

    Args:
        existing_client: Existing httpx client (may be DefaultAsyncHttpxClient)
        network_config: Network configuration to apply

    Returns:
        New client with network config applied, or original client if merge fails
    """
    if network_config is None:
        return existing_client

    fingerprint = network_config_fingerprint(network_config)
    if _get_client_network_fingerprint(existing_client) == fingerprint:
        return existing_client

    network_kwargs = build_network_client_kwargs(network_config)
    if not network_kwargs:
        return existing_client

    try:
        # Extract existing client config (auth, headers)
        existing_config = _extract_client_config(existing_client)

        # Merge headers: existing headers first, then network config (network takes precedence)
        if existing_config.get("headers") and network_kwargs.get("headers"):
            merged_headers = dict(existing_config["headers"])
            merged_headers.update(network_kwargs["headers"])
            network_kwargs["headers"] = merged_headers
        elif existing_config.get("headers"):
            network_kwargs["headers"] = existing_config["headers"]

        # Preserve auth from existing client
        if existing_config.get("auth"):
            network_kwargs["auth"] = existing_config["auth"]

        # Create new client with merged config
        new_client = httpx.AsyncClient(**network_kwargs)

        set_client_network_fingerprint(new_client, fingerprint)

        # If original was DefaultAsyncHttpxClient, wrap the new client
        if isinstance(existing_client, DefaultAsyncHttpxClient):
            wrapped = DefaultAsyncHttpxClient(client=new_client, headers=network_kwargs.get("headers"))  # type: ignore[call-arg]
            set_client_network_fingerprint(wrapped, fingerprint)
            return wrapped

        return new_client
    except Exception as e:
        logger.debug("Could not merge network config into existing http_client, using original client", error=str(e))
        return existing_client


def build_http_client(network_config: NetworkConfig | None) -> dict[str, Any]:
    """
    Build httpx.AsyncClient parameters from network configuration.

    This function creates the appropriate kwargs to pass to httpx.AsyncClient
    based on the provided NetworkConfig.

    Args:
        network_config: Network configuration including TLS, proxy, and timeout settings

    Returns:
        Dictionary of kwargs to pass to httpx.AsyncClient constructor,
        wrapped in {"http_client": AsyncClient(...)} for use with AsyncOpenAI
    """
    network_kwargs = build_network_client_kwargs(network_config)
    if not network_kwargs:
        return {}

    return {"http_client": httpx.AsyncClient(**network_kwargs)}
