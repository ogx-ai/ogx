# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ogx.core.datatypes import (
    FIPS_APPROVED_CIPHERS,
    ServerConfig,
    ServerTLSConfig,
    StackConfig,
)


class TestTLSValidation:
    """FIPS cipher validation at model level; TLS cert enforcement at server startup."""

    def test_default_config_valid(self):
        """Default config without TLS certs should parse without error."""
        config = ServerConfig()
        assert config.tls_certfile is None
        assert config.tls_keyfile is None

    def test_certs_provided_passes(self):
        """Providing both cert and key should pass validation."""
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
        )
        assert config.tls_certfile == "/path/to/cert.pem"

    def test_partial_certs_no_cipher_population(self):
        """Providing only cert (no key) should not auto-populate ciphers."""
        config = ServerConfig(tls_certfile="/path/to/cert.pem")
        assert config.tls_config is None

    def test_insecure_allows_no_certs(self):
        """insecure=True should allow running without TLS certificates."""
        config = ServerConfig(insecure=True)
        assert config.insecure is True
        assert config.tls_certfile is None

    def test_auto_populates_fips_ciphers(self):
        """Should auto-populate FIPS-approved cipher suites when TLS is configured."""
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
        )
        assert config.tls_config is not None
        assert config.tls_config.ciphers == FIPS_APPROVED_CIPHERS

    def test_auto_populates_ciphers_when_tls_config_has_none(self):
        """Should fill in ciphers when tls_config exists but ciphers is None."""
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(),
        )
        assert config.tls_config.ciphers == FIPS_APPROVED_CIPHERS

    def test_preserves_valid_custom_ciphers(self):
        """Should not overwrite explicitly set FIPS-approved ciphers."""
        custom_ciphers = ["ECDHE-RSA-AES256-GCM-SHA384"]
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(ciphers=custom_ciphers),
        )
        assert config.tls_config.ciphers == custom_ciphers

    def test_rejects_non_fips_ciphers(self):
        """Should reject cipher suites not in the FIPS-approved list."""
        with pytest.raises(ValueError, match="FIPS-approved ciphers"):
            ServerConfig(
                tls_certfile="/path/to/cert.pem",
                tls_keyfile="/path/to/key.pem",
                tls_config=ServerTLSConfig(ciphers=["RC4-SHA"]),
            )

    def test_rejects_mixed_ciphers(self):
        """Should reject a list containing any non-FIPS cipher."""
        with pytest.raises(ValueError, match="RC4-SHA"):
            ServerConfig(
                tls_certfile="/path/to/cert.pem",
                tls_keyfile="/path/to/key.pem",
                tls_config=ServerTLSConfig(ciphers=["ECDHE-RSA-AES256-GCM-SHA384", "RC4-SHA"]),
            )

    def test_allows_fips_subset(self):
        """Should accept a subset of FIPS-approved ciphers."""
        subset = ["ECDHE-RSA-AES128-GCM-SHA256", "ECDHE-RSA-AES256-GCM-SHA384"]
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(ciphers=subset),
        )
        assert config.tls_config.ciphers == subset

    def test_rejects_empty_ciphers(self):
        """Should reject an empty cipher list."""
        with pytest.raises(ValueError, match="At least one cipher suite"):
            ServerConfig(
                tls_certfile="/path/to/cert.pem",
                tls_keyfile="/path/to/key.pem",
                tls_config=ServerTLSConfig(ciphers=[]),
            )

    def test_insecure_skips_cipher_validation(self):
        """insecure=True should skip FIPS cipher validation."""
        config = ServerConfig(
            insecure=True,
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
            tls_config=ServerTLSConfig(ciphers=["RC4-SHA"]),
        )
        assert config.tls_config.ciphers == ["RC4-SHA"]

    def test_insecure_does_not_auto_populate_tls_config(self):
        """insecure=True should not auto-create tls_config."""
        config = ServerConfig(insecure=True)
        assert config.tls_config is None

    def test_insecure_default_is_false(self):
        """insecure should default to False."""
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
        )
        assert config.insecure is False


class TestInsecureFlagOverride:
    def test_insecure_in_raw_config(self):
        """insecure=True in raw config dict should set the flag."""
        raw_config = {
            "version": 2,
            "distro_name": "test",
            "providers": {},
            "server": {"insecure": True},
        }
        config = StackConfig(**raw_config)
        assert config.server.insecure is True

    def test_default_config_without_certs_parses(self):
        """Config without TLS certs should parse (enforcement is at server startup)."""
        raw_config = {
            "version": 2,
            "distro_name": "test",
            "providers": {},
        }
        config = StackConfig(**raw_config)
        assert config.server.insecure is False
        assert config.server.tls_certfile is None


class TestVerifyTlsFalse:
    """validate_auth_security should block verify_tls=False unless insecure."""

    def _make_config(self, insecure, verify_tls):
        from ogx.core.datatypes import (
            AuthenticationConfig,
            OAuth2IntrospectionConfig,
            OAuth2TokenAuthConfig,
        )

        server_kwargs = {"insecure": insecure}
        if not insecure:
            server_kwargs["tls_certfile"] = "/path/to/cert.pem"
            server_kwargs["tls_keyfile"] = "/path/to/key.pem"

        server_kwargs["auth"] = AuthenticationConfig(
            provider_config=OAuth2TokenAuthConfig(
                introspection=OAuth2IntrospectionConfig(
                    url="https://auth.example.com/introspect",
                    client_id="client",
                    client_secret="secret",
                ),
                verify_tls=verify_tls,
            ),
        )
        return StackConfig(
            version=2,
            distro_name="test",
            providers={},
            server=server_kwargs,
        )

    def test_rejects_verify_tls_false(self):
        """Should raise SystemExit when verify_tls=False and not insecure."""
        from ogx.core.server.server import validate_auth_security

        config = self._make_config(insecure=False, verify_tls=False)
        with pytest.raises(SystemExit, match="verify_tls=False"):
            validate_auth_security(config)

    def test_insecure_warns_verify_tls_false(self):
        """insecure mode should warn but not crash when verify_tls=False."""
        from ogx.core.server.server import validate_auth_security

        config = self._make_config(insecure=True, verify_tls=False)
        with patch("ogx.core.server.server.logger") as mock_logger:
            validate_auth_security(config)
            mock_logger.warning.assert_called_once()

    def test_verify_tls_true_passes(self):
        """verify_tls=True should pass without error."""
        from ogx.core.server.server import validate_auth_security

        config = self._make_config(insecure=False, verify_tls=True)
        validate_auth_security(config)


class TestHSTSMiddleware:
    def test_hsts_header_default_max_age(self):
        """HSTS header should use default max-age of 1 year."""
        from ogx.core.server.server import HSTSMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(HSTSMiddleware)
        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200
        assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"

    def test_hsts_header_custom_max_age(self):
        """HSTS header should respect custom max-age."""
        from ogx.core.server.server import HSTSMiddleware

        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(HSTSMiddleware, max_age=86400)
        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200
        assert response.headers["strict-transport-security"] == "max-age=86400; includeSubDomains"

    def test_hsts_header_absent_without_middleware(self):
        """HSTS header should not be present when middleware is not added."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200
        assert "strict-transport-security" not in response.headers

    def test_hsts_max_age_config_default(self):
        """ServerConfig hsts_max_age should default to 1 year."""
        config = ServerConfig(
            tls_certfile="/path/to/cert.pem",
            tls_keyfile="/path/to/key.pem",
        )
        assert config.hsts_max_age == 31536000

    def test_hsts_max_age_config_zero_disables(self):
        """hsts_max_age=0 should be valid (used to disable HSTS)."""
        config = ServerConfig(
            insecure=True,
            hsts_max_age=0,
        )
        assert config.hsts_max_age == 0

    def test_hsts_max_age_config_rejects_negative(self):
        """hsts_max_age must not be negative."""
        with pytest.raises(ValueError):
            ServerConfig(insecure=True, hsts_max_age=-1)


class TestFIPSCipherConstants:
    def test_fips_ciphers_are_aes_gcm_only(self):
        """All FIPS-approved ciphers should be AES-GCM variants."""
        for cipher in FIPS_APPROVED_CIPHERS:
            assert "GCM" in cipher, f"Cipher {cipher} is not AES-GCM"

    def test_fips_ciphers_no_chacha(self):
        """CHACHA20-POLY1305 should not be in FIPS-approved list."""
        for cipher in FIPS_APPROVED_CIPHERS:
            assert "CHACHA" not in cipher, f"Cipher {cipher} should not be in FIPS list"

    def test_server_tls_config_defaults(self):
        """ServerTLSConfig should have sensible defaults."""
        config = ServerTLSConfig()
        assert config.ciphers is None
