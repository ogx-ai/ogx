# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import datetime
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
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


def generate_self_signed_cert(base_dir: Path, domain: str = "localhost") -> tuple[Path, Path]:
    """Generate a self-signed TLS certificate and key for local development."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OGX"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(domain), x509.DNSName("*")]), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_path = base_dir / "server.crt"
    key_path = base_dir / "server.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path
