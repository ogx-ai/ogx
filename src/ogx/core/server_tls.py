# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

FIPS_APPROVED_CIPHERS = [
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES128-GCM-SHA256",
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "DHE-RSA-AES128-GCM-SHA256",
    "DHE-RSA-AES256-GCM-SHA384",
]


class ServerTLSConfig(BaseModel):
    """TLS cipher suite configuration for the server."""

    # Note: minimum TLS version is not configurable here because uvicorn does not
    # expose ssl.SSLContext.minimum_version. Python 3.10+ defaults to TLS 1.2 minimum.
    ciphers: list[str] | None = Field(
        default=None,
        description="Allowed TLS 1.2 cipher suites (OpenSSL names). Defaults to FIPS-approved AES-GCM ciphers.",
    )


def validate_fips_tls(
    insecure: bool,
    tls_certfile: str | None,
    tls_keyfile: str | None,
    tls_config: ServerTLSConfig | None,
) -> ServerTLSConfig | None:
    """Apply FIPS cipher defaults and validate cipher suites when TLS is configured."""
    if insecure or not (tls_certfile and tls_keyfile):
        return tls_config
    if tls_config is None:
        return ServerTLSConfig(ciphers=FIPS_APPROVED_CIPHERS)
    if tls_config.ciphers is None:
        tls_config.ciphers = FIPS_APPROVED_CIPHERS
    elif not tls_config.ciphers:
        raise ValueError("At least one cipher suite must be specified.")
    elif invalid := set(tls_config.ciphers) - set(FIPS_APPROVED_CIPHERS):
        raise ValueError(f"FIPS-approved ciphers required. Invalid: {sorted(invalid)}")
    return tls_config
