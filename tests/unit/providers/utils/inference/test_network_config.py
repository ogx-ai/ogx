# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
import tempfile
from pathlib import Path

import httpx
import pytest

from ogx.providers.utils.inference.http_client import (
    _build_proxy_mounts,
    _build_ssl_context,
    build_http_client,
)
from ogx.providers.utils.inference.http_client import (
    build_network_client_kwargs as _build_network_client_kwargs,
)
from ogx.providers.utils.inference.model_registry import (
    NetworkConfig,
    ProxyConfig,
    TimeoutConfig,
    TLSConfig,
)


class TestTLSConfig:
    """Tests for TLSConfig model validation."""

    def test_default_values(self):
        """Test TLSConfig with default values."""
        config = TLSConfig()
        assert config.verify is True
        assert config.min_version is None
        assert config.ciphers is None
        assert config.client_cert is None
        assert config.client_key is None

    def test_verify_boolean(self):
        """Test TLSConfig with boolean verify."""
        config = TLSConfig(verify=False)
        assert config.verify is False

    def test_verify_valid_path(self):
        """Test TLSConfig with valid certificate path."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert content")
            cert_path = f.name

        try:
            config = TLSConfig(verify=cert_path)
            assert config.verify.resolve() == Path(cert_path).resolve()
        finally:
            Path(cert_path).unlink()

    def test_verify_invalid_path(self):
        """Test TLSConfig with invalid certificate path raises error."""
        with pytest.raises(ValueError, match="TLS certificate file does not exist"):
            TLSConfig(verify="/nonexistent/path/to/cert.pem")

    def test_min_version_valid(self):
        """Test TLSConfig with valid min_version values."""
        config_12 = TLSConfig(min_version="TLSv1.2")
        assert config_12.min_version == "TLSv1.2"

        config_13 = TLSConfig(min_version="TLSv1.3")
        assert config_13.min_version == "TLSv1.3"

    def test_ciphers_list(self):
        """Test TLSConfig with cipher list."""
        ciphers = ["ECDHE+AESGCM", "DHE+AESGCM"]
        config = TLSConfig(ciphers=ciphers)
        assert config.ciphers == ciphers

    def test_mtls_requires_both_cert_and_key(self):
        """Test that mTLS requires both client_cert and client_key."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert")
            cert_path = f.name

        try:
            with pytest.raises(ValueError, match="Both client_cert and client_key must be provided"):
                TLSConfig(client_cert=cert_path)
        finally:
            Path(cert_path).unlink()

    def test_mtls_valid_paths(self):
        """Test TLSConfig with valid mTLS paths."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as cert_file:
            cert_file.write(b"fake cert")
            cert_path = cert_file.name

        with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as key_file:
            key_file.write(b"fake key")
            key_path = key_file.name

        try:
            config = TLSConfig(client_cert=cert_path, client_key=key_path)
            assert config.client_cert.resolve() == Path(cert_path).resolve()
            assert config.client_key.resolve() == Path(key_path).resolve()
        finally:
            Path(cert_path).unlink()
            Path(key_path).unlink()


class TestProxyConfig:
    """Tests for ProxyConfig model validation."""

    def test_default_values(self):
        """Test ProxyConfig with default values."""
        config = ProxyConfig()
        assert config.url is None
        assert config.http is None
        assert config.https is None
        assert config.no_proxy is None

    def test_single_proxy_url(self):
        """Test ProxyConfig with single proxy URL."""
        config = ProxyConfig(url="http://proxy.example.com:8080")
        assert str(config.url) == "http://proxy.example.com:8080/"

    def test_granular_proxy_settings(self):
        """Test ProxyConfig with HTTP and HTTPS proxies."""
        config = ProxyConfig(http="http://proxy:8080", https="https://proxy:8443")
        assert str(config.http) == "http://proxy:8080/"
        assert str(config.https) == "https://proxy:8443/"

    def test_no_proxy_list(self):
        """Test ProxyConfig with no_proxy list."""
        config = ProxyConfig(no_proxy=["localhost", "127.0.0.1", ".internal.corp"])
        assert config.no_proxy == ["localhost", "127.0.0.1", ".internal.corp"]

    def test_cacert_valid_path(self):
        """Test ProxyConfig with valid CA certificate path."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert content")
            cert_path = f.name

        try:
            config = ProxyConfig(url="http://proxy:8080", cacert=cert_path)
            assert config.cacert.resolve() == Path(cert_path).resolve()
        finally:
            Path(cert_path).unlink()

    def test_cacert_invalid_path(self):
        """Test ProxyConfig with invalid CA certificate path raises error."""
        with pytest.raises(ValueError, match="Proxy CA certificate file does not exist"):
            ProxyConfig(url="http://proxy:8080", cacert="/nonexistent/path/to/cert.pem")

    def test_cacert_with_proxy_url(self):
        """Test ProxyConfig with CA cert and proxy URL."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert")
            cert_path = f.name

        try:
            config = ProxyConfig(url="http://proxy:8080", cacert=cert_path)
            assert str(config.url) == "http://proxy:8080/"
            assert config.cacert.resolve() == Path(cert_path).resolve()
        finally:
            Path(cert_path).unlink()

    def test_cacert_with_granular_proxies(self):
        """Test ProxyConfig with CA cert and granular proxy settings."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert")
            cert_path = f.name

        try:
            config = ProxyConfig(http="http://proxy:8080", https="https://proxy:8443", cacert=cert_path)
            assert str(config.http) == "http://proxy:8080/"
            assert str(config.https) == "https://proxy:8443/"
            assert config.cacert.resolve() == Path(cert_path).resolve()
        finally:
            Path(cert_path).unlink()

    def test_url_and_granular_conflict(self):
        """Test that url and http/https cannot be specified together."""
        with pytest.raises(ValueError, match="Cannot specify both 'url' and 'http'/'https'"):
            ProxyConfig(url="http://proxy:8080", http="http://other:8080")


class TestTimeoutConfig:
    """Tests for TimeoutConfig model."""

    def test_default_values(self):
        """Test TimeoutConfig with default values."""
        config = TimeoutConfig()
        assert config.connect is None
        assert config.read is None

    def test_with_connect_only(self):
        """Test TimeoutConfig with connect timeout only."""
        config = TimeoutConfig(connect=5.0)
        assert config.connect == 5.0
        assert config.read is None

    def test_with_read_only(self):
        """Test TimeoutConfig with read timeout only."""
        config = TimeoutConfig(read=30.0)
        assert config.connect is None
        assert config.read == 30.0

    def test_with_both(self):
        """Test TimeoutConfig with both connect and read timeouts."""
        config = TimeoutConfig(connect=5.0, read=30.0)
        assert config.connect == 5.0
        assert config.read == 30.0


class TestNetworkConfig:
    """Tests for NetworkConfig model."""

    def test_default_values(self):
        """Test NetworkConfig with default values."""
        config = NetworkConfig()
        assert config.tls is None
        assert config.proxy is None
        assert config.timeout is None

    def test_with_tls_config(self):
        """Test NetworkConfig with TLS configuration."""
        config = NetworkConfig(tls=TLSConfig(verify=False))
        assert config.tls is not None
        assert config.tls.verify is False

    def test_with_proxy_config(self):
        """Test NetworkConfig with proxy configuration."""
        config = NetworkConfig(proxy=ProxyConfig(url="http://proxy:8080"))
        assert config.proxy is not None
        assert str(config.proxy.url) == "http://proxy:8080/"

    def test_with_timeout_float(self):
        """Test NetworkConfig with timeout as float."""
        config = NetworkConfig(timeout=30.0)
        assert config.timeout == 30.0

    def test_with_timeout_config(self):
        """Test NetworkConfig with TimeoutConfig."""
        timeout_config = TimeoutConfig(connect=5.0, read=30.0)
        config = NetworkConfig(timeout=timeout_config)
        assert isinstance(config.timeout, TimeoutConfig)
        assert config.timeout.connect == 5.0
        assert config.timeout.read == 30.0

    def test_full_config(self):
        """Test NetworkConfig with all options."""
        config = NetworkConfig(
            tls=TLSConfig(verify=True, min_version="TLSv1.2"),
            proxy=ProxyConfig(url="http://proxy:8080"),
            timeout=60.0,
        )
        assert config.tls.verify is True
        assert config.tls.min_version == "TLSv1.2"
        assert str(config.proxy.url) == "http://proxy:8080/"
        assert config.timeout == 60.0


class TestBuildSSLContext:
    """Tests for _build_ssl_context function."""

    def test_simple_verify_true(self):
        """Test SSL context with simple verify=True returns boolean."""
        tls_config = TLSConfig(verify=True)
        result = _build_ssl_context(tls_config)
        assert result is True

    def test_simple_verify_false(self):
        """Test SSL context with simple verify=False returns boolean."""
        tls_config = TLSConfig(verify=False)
        result = _build_ssl_context(tls_config)
        assert result is False

    def test_verify_with_path(self):
        """Test SSL context with CA path returns a string (httpx requires str, not Path)."""
        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert")
            cert_path = f.name

        try:
            tls_config = TLSConfig(verify=cert_path)
            result = _build_ssl_context(tls_config)
            assert isinstance(result, str)
            assert Path(result).resolve() == Path(cert_path).resolve()
        finally:
            Path(cert_path).unlink()

    def test_with_min_version_returns_ssl_context(self):
        """Test that min_version creates an SSL context."""
        tls_config = TLSConfig(min_version="TLSv1.2")
        result = _build_ssl_context(tls_config)
        assert isinstance(result, ssl.SSLContext)
        assert result.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_with_min_version_tls13(self):
        """Test SSL context with TLSv1.3."""
        tls_config = TLSConfig(min_version="TLSv1.3")
        result = _build_ssl_context(tls_config)
        assert isinstance(result, ssl.SSLContext)
        assert result.minimum_version == ssl.TLSVersion.TLSv1_3

    def test_with_ciphers_returns_ssl_context(self):
        """Test that ciphers create an SSL context."""
        tls_config = TLSConfig(ciphers=["ECDHE+AESGCM"])
        result = _build_ssl_context(tls_config)
        assert isinstance(result, ssl.SSLContext)

    def test_verify_false_with_advanced_options(self):
        """Test SSL context with verify=False and advanced options."""
        tls_config = TLSConfig(verify=False, min_version="TLSv1.2")
        result = _build_ssl_context(tls_config)
        assert isinstance(result, ssl.SSLContext)
        assert result.verify_mode == ssl.CERT_NONE
        assert result.check_hostname is False


class TestBuildProxyMounts:
    """Tests for _build_proxy_mounts function."""

    def test_none_config(self):
        """Test proxy mounts with empty config."""
        config = ProxyConfig()
        result = _build_proxy_mounts(config)
        assert result is None

    def test_single_url(self):
        """Test proxy mounts with single URL."""
        config = ProxyConfig(url="http://proxy:8080")
        result = _build_proxy_mounts(config)
        assert result is not None
        assert "http://" in result
        assert "https://" in result

    def test_granular_proxies(self):
        """Test proxy mounts with granular settings."""
        config = ProxyConfig(http="http://proxy:8080", https="https://proxy:8443")
        result = _build_proxy_mounts(config)
        assert result is not None
        assert "http://" in result
        assert "https://" in result

    def test_proxy_with_cacert(self):
        """Test proxy mounts with CA certificate configuration."""
        import ssl

        # Try to use system CA bundle as a valid cert file for testing
        system_ca_bundle = ssl.get_default_verify_paths().cafile
        if system_ca_bundle and Path(system_ca_bundle).exists():
            cert_path = system_ca_bundle
        else:
            # Create a dummy file - httpx will try to load it but we'll catch the error
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt", delete=False) as f:
                f.write(b"dummy cert for testing")
                cert_path = f.name

        try:
            config = ProxyConfig(url="http://proxy:8080", cacert=cert_path)
            # Verify config accepts the cert path (compare resolved paths)
            assert config.cacert.resolve() == Path(cert_path).resolve()
            # The transport creation may fail with invalid cert content, but that's expected
            # We verify the config structure is correct and that verify is set in transport_kwargs
            try:
                result = _build_proxy_mounts(config)
                assert result is not None
                assert "http://" in result
                assert "https://" in result
                assert isinstance(result["http://"], httpx.AsyncHTTPTransport)
            except ssl.SSLError:
                # Expected for dummy cert files - the important part is that the config was accepted
                pass
        finally:
            # Only delete if we created a temp file
            if not (system_ca_bundle and Path(system_ca_bundle).exists()):
                if Path(cert_path).exists():
                    Path(cert_path).unlink()

    def test_granular_proxies_with_cacert(self):
        """Test granular proxy mounts with CA certificate configuration."""
        import ssl

        # Try to use system CA bundle if available
        system_ca_bundle = ssl.get_default_verify_paths().cafile
        if system_ca_bundle and Path(system_ca_bundle).exists():
            cert_path = system_ca_bundle
        else:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt", delete=False) as f:
                f.write(b"dummy cert for testing")
                cert_path = f.name

        try:
            config = ProxyConfig(http="http://proxy:8080", https="https://proxy:8443", cacert=cert_path)
            assert config.cacert.resolve() == Path(cert_path).resolve()
            # The transport creation may fail with invalid cert content, but that's expected
            # We verify the config structure is correct
            result = None
            try:
                result = _build_proxy_mounts(config)
                assert result is not None
                assert "http://" in result
                assert "https://" in result
                assert isinstance(result["http://"], httpx.AsyncHTTPTransport)
            except ssl.SSLError:
                # Expected for dummy cert files - the important part is that the config was accepted
                pass
            if result is not None:
                assert isinstance(result["https://"], httpx.AsyncHTTPTransport)
        finally:
            if not (system_ca_bundle and Path(system_ca_bundle).exists()):
                if Path(cert_path).exists():
                    Path(cert_path).unlink()


class TestBuildHttpClient:
    """Tests for build_http_client function."""

    def test_none_config(self):
        """Test that None config returns empty dict."""
        result = build_http_client(None)
        assert result == {}

    def test_empty_network_config(self):
        """Test that empty network config returns empty dict."""
        config = NetworkConfig()
        result = build_http_client(config)
        assert result == {}

    def test_with_tls_config(self):
        """Test http client with TLS config."""
        config = NetworkConfig(tls=TLSConfig(verify=False))
        result = build_http_client(config)
        assert "http_client" in result
        assert isinstance(result["http_client"], httpx.AsyncClient)

    def test_with_timeout_float(self):
        """Test http client with timeout as float."""
        config = NetworkConfig(timeout=30.0)
        result = build_http_client(config)
        assert "http_client" in result
        assert isinstance(result["http_client"], httpx.AsyncClient)
        # Verify timeout is set
        client_kwargs = _build_network_client_kwargs(config)
        assert "timeout" in client_kwargs
        assert isinstance(client_kwargs["timeout"], httpx.Timeout)

    def test_with_timeout_config(self):
        """Test http client with TimeoutConfig."""
        timeout_config = TimeoutConfig(connect=5.0, read=30.0)
        config = NetworkConfig(timeout=timeout_config)
        result = build_http_client(config)
        assert "http_client" in result
        assert isinstance(result["http_client"], httpx.AsyncClient)
        # Verify timeout is set with both connect and read
        client_kwargs = _build_network_client_kwargs(config)
        assert "timeout" in client_kwargs
        timeout = client_kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 5.0
        assert timeout.read == 30.0

    def test_with_timeout_config_connect_only(self):
        """Test http client with TimeoutConfig having only connect."""
        timeout_config = TimeoutConfig(connect=5.0)
        config = NetworkConfig(timeout=timeout_config)
        client_kwargs = _build_network_client_kwargs(config)
        assert "timeout" in client_kwargs
        timeout = client_kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 5.0

    def test_with_timeout_config_read_only(self):
        """Test http client with TimeoutConfig having only read."""
        timeout_config = TimeoutConfig(read=30.0)
        config = NetworkConfig(timeout=timeout_config)
        client_kwargs = _build_network_client_kwargs(config)
        assert "timeout" in client_kwargs
        timeout = client_kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 30.0

    def test_with_proxy(self):
        """Test http client with proxy config."""
        config = NetworkConfig(proxy=ProxyConfig(url="http://proxy:8080"))
        result = build_http_client(config)
        assert "http_client" in result
        assert isinstance(result["http_client"], httpx.AsyncClient)


class TestVLLMBackwardCompatibility:
    """Tests for vLLM backward compatibility with tls_verify."""

    def test_legacy_tls_verify_true(self):
        """Test that legacy tls_verify=True is migrated."""
        from ogx.providers.remote.inference.vllm import VLLMInferenceAdapterConfig

        with pytest.warns(DeprecationWarning, match="tls_verify.*deprecated"):
            config = VLLMInferenceAdapterConfig(tls_verify=True)

        assert config.network is not None
        assert config.network.tls is not None
        assert config.network.tls.verify is True

    def test_legacy_tls_verify_false(self):
        """Test that legacy tls_verify=False is migrated."""
        from ogx.providers.remote.inference.vllm import VLLMInferenceAdapterConfig

        with pytest.warns(DeprecationWarning, match="tls_verify.*deprecated"):
            config = VLLMInferenceAdapterConfig(tls_verify=False)

        assert config.network is not None
        assert config.network.tls is not None
        assert config.network.tls.verify is False

    def test_legacy_tls_verify_path(self):
        """Test that legacy tls_verify path is migrated."""
        from ogx.providers.remote.inference.vllm import VLLMInferenceAdapterConfig

        with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as f:
            f.write(b"fake cert")
            cert_path = f.name

        try:
            with pytest.warns(DeprecationWarning, match="tls_verify.*deprecated"):
                config = VLLMInferenceAdapterConfig(tls_verify=cert_path)

            assert config.network is not None
            assert config.network.tls is not None
            assert config.network.tls.verify.resolve() == Path(cert_path).resolve()
        finally:
            Path(cert_path).unlink()

    def test_new_network_config_style(self):
        """Test that new network config style works."""
        from ogx.providers.remote.inference.vllm import VLLMInferenceAdapterConfig

        config = VLLMInferenceAdapterConfig(
            network=NetworkConfig(
                tls=TLSConfig(verify=True, min_version="TLSv1.2"),
                timeout=30.0,
            )
        )
        assert config.network is not None
        assert config.network.tls.verify is True
        assert config.network.tls.min_version == "TLSv1.2"
        assert config.network.timeout == 30.0

    def test_network_not_overwritten_by_tls_verify(self):
        """Test that existing network.tls is not overwritten by tls_verify."""
        from ogx.providers.remote.inference.vllm import VLLMInferenceAdapterConfig

        with pytest.warns(DeprecationWarning, match="tls_verify.*deprecated"):
            config = VLLMInferenceAdapterConfig(
                tls_verify=False,
                network=NetworkConfig(tls=TLSConfig(verify=True, min_version="TLSv1.3")),
            )

        # network.tls should be preserved since it was explicitly set
        assert config.network.tls.verify is True
        assert config.network.tls.min_version == "TLSv1.3"
