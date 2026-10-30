# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the connect TLS helpers."""

import ssl

from ogx.cli.connect.tls import is_untrusted_certificate_error, untrusted_certificate_client

from .connect_helpers import make_untrusted_certificate_error


class TestIsUntrustedCertificateError:
    def test_detects_cert_error_in_context_chain(self) -> None:
        assert is_untrusted_certificate_error(make_untrusted_certificate_error()) is True

    def test_detects_bare_cert_error(self) -> None:
        assert is_untrusted_certificate_error(ssl.SSLCertVerificationError("self-signed certificate")) is True

    def test_detects_cert_error_in_cause_chain(self) -> None:
        ssl_error = ssl.SSLCertVerificationError("self-signed certificate")
        wrapper = Exception("connection failed")
        try:
            raise wrapper from ssl_error
        except Exception as e:
            assert is_untrusted_certificate_error(e) is True

    def test_plain_connection_error_is_not_cert_error(self) -> None:
        refused = ConnectionRefusedError("Connection refused")
        wrapper = Exception("connection failed")
        wrapper.__context__ = refused
        assert is_untrusted_certificate_error(wrapper) is False


class TestUntrustedCertificateClient:
    def test_disables_certificate_verification(self) -> None:
        client = untrusted_certificate_client()
        assert client._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE
        client.close()
