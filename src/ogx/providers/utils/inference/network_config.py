# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class TLSConfig(BaseModel):
    """TLS/SSL configuration for secure connections."""

    verify: bool | Path = Field(
        default=True,
        description="Whether to verify TLS certificates. Can be a boolean or a path to a CA certificate file.",
    )
    min_version: Literal["TLSv1.2", "TLSv1.3"] | None = Field(
        default=None,
        description="Minimum TLS version to use. Defaults to system default if not specified.",
    )
    ciphers: list[str] | None = Field(
        default=None,
        description="List of allowed cipher suites (e.g., ['ECDHE+AESGCM', 'DHE+AESGCM']).",
    )
    client_cert: Path | None = Field(
        default=None,
        description="Path to client certificate file for mTLS authentication.",
    )
    client_key: Path | None = Field(
        default=None,
        description="Path to client private key file for mTLS authentication.",
    )

    @field_validator("verify", mode="before")
    @classmethod
    def validate_verify(cls, v: bool | str | Path) -> bool | Path:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            cert_path = Path(v).expanduser().resolve()
        else:
            cert_path = v.expanduser().resolve()
        if not cert_path.exists():
            raise ValueError(f"TLS certificate file does not exist: {v}")
        if not cert_path.is_file():
            raise ValueError(f"Certificate path is not a file: {v}")
        return cert_path

    @field_validator("client_cert", "client_key", mode="before")
    @classmethod
    def validate_cert_paths(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        if isinstance(v, str):
            cert_path = Path(v).expanduser().resolve()
        else:
            cert_path = v.expanduser().resolve()
        if not cert_path.exists():
            raise ValueError(f"Certificate/key file does not exist: {v}")
        if not cert_path.is_file():
            raise ValueError(f"Certificate key path is not a file: {v}")
        return cert_path

    @model_validator(mode="after")
    def validate_mtls_pair(self) -> "TLSConfig":
        if (self.client_cert is None) != (self.client_key is None):
            raise ValueError("Both client_cert and client_key must be provided together for mTLS")
        return self


class ProxyConfig(BaseModel):
    """Proxy configuration for HTTP connections."""

    url: HttpUrl | None = Field(
        default=None,
        description="Single proxy URL for all connections (e.g., 'http://proxy.example.com:8080').",
    )
    http: HttpUrl | None = Field(
        default=None,
        description="Proxy URL for HTTP connections.",
    )
    https: HttpUrl | None = Field(
        default=None,
        description="Proxy URL for HTTPS connections.",
    )
    cacert: Path | None = Field(
        default=None,
        description="Path to CA certificate file for verifying the proxy's certificate. Required for proxies in interception mode.",
    )
    no_proxy: list[str] | None = Field(
        default=None,
        description="List of hosts that should bypass the proxy (e.g., ['localhost', '127.0.0.1', '.internal.corp']).",
    )

    @field_validator("cacert", mode="before")
    @classmethod
    def validate_cacert(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        if isinstance(v, str):
            cert_path = Path(v).expanduser().resolve()
        else:
            cert_path = v.expanduser().resolve()
        if not cert_path.exists():
            raise ValueError(f"Proxy CA certificate file does not exist: {v}")
        if not cert_path.is_file():
            raise ValueError(f"Proxy CA certificate path is not a file: {v}")
        return cert_path

    @model_validator(mode="after")
    def validate_proxy_config(self) -> "ProxyConfig":
        if self.url and (self.http or self.https):
            raise ValueError("Cannot specify both 'url' and 'http'/'https' proxy settings")
        return self


class TimeoutConfig(BaseModel):
    """Timeout configuration for HTTP connections."""

    connect: float | None = Field(
        default=None,
        description="Connection timeout in seconds.",
    )
    read: float | None = Field(
        default=None,
        description="Read timeout in seconds.",
    )


class LimitsConfig(BaseModel):
    """HTTP connection pool limits, matching httpx.Limits.

    Defaults match httpx's own defaults, so setting one field doesn't leave the others
    unbounded.
    """

    max_connections: int | None = Field(
        default=100,
        ge=1,
        description="Maximum number of concurrent connections in the pool. None means no limit. Values must be >= 1 if set.",
    )
    max_keepalive_connections: int | None = Field(
        default=20,
        ge=0,
        description="Maximum number of idle keep-alive connections to retain. None means no limit. Values must be >= 0 if set.",
    )
    keepalive_expiry: float | None = Field(
        default=5.0,
        ge=0,
        description="Time in seconds to keep idle keep-alive connections open before closing them. None means no expiry. Values must be >= 0 if set.",
    )


class NetworkConfig(BaseModel):
    """Network configuration for remote provider connections."""

    tls: TLSConfig | None = Field(
        default=None,
        description="TLS/SSL configuration for secure connections.",
    )
    proxy: ProxyConfig | None = Field(
        default=None,
        description="Proxy configuration for HTTP connections.",
    )
    timeout: float | TimeoutConfig | None = Field(
        default=None,
        description="Timeout configuration. Can be a float (for both connect and read) or a TimeoutConfig object with separate connect and read timeouts.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Additional HTTP headers to include in all requests.",
    )
    limits: LimitsConfig | None = Field(
        default=None,
        description="HTTP connection pool limits (max connections, keepalive connections, keepalive expiry). None uses httpx's own defaults.",
    )
