# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for Bedrock SigV4 authentication.

These tests verify:
1. SigV4 auth handler correctly signs requests
2. Auth mode detection (bearer vs SigV4)
3. Credential chain integration
4. Error handling
"""

# Check if boto3 is available for SigV4 tests
import importlib.util
from unittest.mock import MagicMock, patch

import httpx
import pytest

HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestBedrockSigV4Auth:
    """Tests for BedrockSigV4Auth httpx.Auth implementation."""

    def test_auth_flow_signs_request(self):
        """SigV4 auth should add AWS signature headers to request."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        # Mock boto3 credentials
        mock_creds = MagicMock()
        mock_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_creds.token = None

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = mock_creds.access_key
        mock_frozen_creds.secret_key = mock_creds.secret_key
        mock_frozen_creds.token = mock_creds.token

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-1", service="bedrock")

            # Create a test request
            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                headers={"content-type": "application/json"},
                content=b'{"model": "test"}',
            )

            # Run auth flow
            gen = auth.auth_flow(request)
            signed_request = next(gen)

            # Verify SigV4 headers were added
            assert "authorization" in signed_request.headers
            assert "x-amz-date" in signed_request.headers
            assert "AWS4-HMAC-SHA256" in signed_request.headers["authorization"]

    def test_auth_flow_with_explicit_role_assumption(self):
        """SigV4 auth should use RefreshableBotoSession when role_arn is provided."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "ASIAEXP_ROLE_KEY"
        mock_frozen_creds.secret_key = "exp_secret"
        mock_frozen_creds.token = "exp_token"

        with patch(
            "ogx.providers.utils.bedrock.refreshable_boto_session.RefreshableBotoSession"
        ) as mock_refreshable_cls:
            mock_refreshable = MagicMock()
            mock_refreshable_cls.return_value = mock_refreshable
            mock_session = MagicMock()
            mock_refreshable.refreshable_session.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(
                region="us-east-1",
                aws_role_arn="arn:aws:iam::123456789012:role/test-role",
                aws_web_identity_token_file="/path/to/token",
                aws_role_session_name="test-session",
            )

            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                content=b"{}",
            )

            gen = auth.auth_flow(request)
            signed_request = next(gen)

            # Verify RefreshableBotoSession was called with correct args
            mock_refreshable_cls.assert_called_once_with(
                region_name="us-east-1",
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=None,
                profile_name=None,
                sts_arn="arn:aws:iam::123456789012:role/test-role",
                web_identity_token_file="/path/to/token",
                session_name="test-session",
                session_ttl=3600,
            )
            assert signed_request.headers["x-amz-security-token"] == "exp_token"

    def test_auth_flow_with_session_token(self):
        """SigV4 auth should include X-Amz-Security-Token for STS credentials."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "ASIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = "FwoGZXIvYXdzEBYaDG..."  # STS session token

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-west-2", service="bedrock")

            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
                headers={"content-type": "application/json"},
                content=b'{"model": "test"}',
            )

            gen = auth.auth_flow(request)
            signed_request = next(gen)

            # Verify session token header is present
            assert "x-amz-security-token" in signed_request.headers
            assert signed_request.headers["x-amz-security-token"] == mock_frozen_creds.token

    def test_auth_raises_on_missing_credentials(self):
        """SigV4 auth should raise clear error when credentials unavailable."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value = None

            auth = BedrockSigV4Auth(region="us-east-1")

            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                content=b"{}",
            )

            with pytest.raises(RuntimeError, match="Failed to load AWS credentials"):
                gen = auth.auth_flow(request)
                next(gen)


class TestBedrockConfigAuthDetection:
    """Tests for BedrockConfig auth mode detection."""

    def test_has_bearer_token_with_token(self):
        """Config should detect when bearer token is present."""
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        # Use api_key as that's the alias for auth_credential
        config = BedrockConfig(api_key="my-bearer-token")
        assert config.has_bearer_token() is True

    def test_has_bearer_token_without_token(self):
        """Config should detect when bearer token is absent."""
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig()
        assert config.has_bearer_token() is False

    def test_has_bearer_token_with_empty_string(self):
        """Empty string should be treated as no token."""
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(api_key="")
        assert config.has_bearer_token() is False

    def test_has_bearer_token_with_whitespace(self):
        """Whitespace-only string should be treated as no token."""
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(api_key="   ")
        assert config.has_bearer_token() is False


class TestBedrockInferenceAdapterAuthMode:
    """Tests for BedrockInferenceAdapter auth mode selection."""

    def test_should_use_sigv4_when_no_bearer_token(self):
        """Adapter should use SigV4 when no bearer token configured."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        # Mock get_request_provider_data to return None
        with patch.object(adapter, "get_request_provider_data", return_value=None):
            assert adapter._should_use_sigv4() is True

    def test_should_not_use_sigv4_when_bearer_token_in_config(self):
        """Adapter should use bearer auth when token in config."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(
            region_name="us-east-1",
            api_key="my-bearer-token",  # Use api_key alias
        )
        adapter = BedrockInferenceAdapter(config=config)

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            assert adapter._should_use_sigv4() is False

    def test_should_not_use_sigv4_when_bearer_token_in_provider_data(self):
        """Adapter should use bearer auth when token in provider data."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import (
            BedrockConfig,
            BedrockProviderDataValidator,
        )

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        provider_data = BedrockProviderDataValidator(aws_bearer_token_bedrock="per-request-token")
        with patch.object(adapter, "get_request_provider_data", return_value=provider_data):
            assert adapter._should_use_sigv4() is False

    def test_get_extra_client_params_skips_sigv4_client_when_bearer_override(self):
        """Per-request bearer token override must not be silently discarded by the SigV4 client.

        When the server starts in SigV4 mode (_sigv4_http_client is not None) but a request
        arrives with aws_bearer_token_bedrock in provider data, get_extra_client_params()
        must return {} so the OpenAI SDK uses the bearer token instead of SigV4 auth.
        """
        from unittest.mock import MagicMock

        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import (
            BedrockConfig,
            BedrockProviderDataValidator,
        )

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        # Simulate that initialize() already built the SigV4 client
        adapter._sigv4_http_client = MagicMock()

        # Per-request bearer token override in provider data
        provider_data = BedrockProviderDataValidator(aws_bearer_token_bedrock="per-request-token")
        with patch.object(adapter, "get_request_provider_data", return_value=provider_data):
            params = adapter.get_extra_client_params()
            # Must return {} — the bearer token path must not receive the SigV4 http_client,
            # which would strip and replace the Authorization header
            assert params == {}

    def test_get_extra_client_params_uses_sigv4_client_when_no_override(self):
        """SigV4 client is returned when no per-request bearer token is present."""
        from unittest.mock import MagicMock

        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)
        mock_client = MagicMock()
        adapter._sigv4_http_client = mock_client

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            params = adapter.get_extra_client_params()
            assert params == {"http_client": mock_client}

    def test_should_use_sigv4_when_provider_data_token_is_whitespace(self):
        """Adapter should use SigV4 when provider data token is whitespace-only."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import (
            BedrockConfig,
            BedrockProviderDataValidator,
        )

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        # Whitespace-only token should be treated as no token (use SigV4)
        provider_data = BedrockProviderDataValidator(aws_bearer_token_bedrock="   ")
        with patch.object(adapter, "get_request_provider_data", return_value=provider_data):
            assert adapter._should_use_sigv4() is True

    def test_get_api_key_returns_placeholder_for_sigv4(self):
        """When using SigV4, get_api_key should return placeholder to satisfy OpenAIMixin validation."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            api_key = adapter.get_api_key()
            # Placeholder satisfies OpenAIMixin validation; SigV4 auth handler replaces
            # the Bearer header with proper SigV4 signature (OCI pattern)
            assert api_key == "<NOTUSED>"

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    def test_client_uses_sigv4_auth_when_no_bearer_token(self):
        """_build_sigv4_http_client should use correct service name and pass config fields."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-west-2")
        adapter = BedrockInferenceAdapter(config=config)

        with patch("ogx.providers.utils.bedrock.sigv4_auth.BedrockSigV4Auth") as mock_auth_cls:
            mock_auth_cls.return_value = MagicMock()
            adapter._build_sigv4_http_client()

            # Verify auth was created with correct service name ("bedrock", not "bedrock-runtime")
            call_kwargs = mock_auth_cls.call_args[1]
            assert call_kwargs["region"] == "us-west-2"
            assert call_kwargs["service"] == "bedrock"

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    def test_sigv4_http_client_cached_after_initialize(self):
        """_sigv4_http_client should be created once in initialize() and reused."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        with patch.object(adapter, "_build_sigv4_http_client") as mock_build:
            mock_build.return_value = MagicMock()

            # Simulate initialize() — called once
            adapter._sigv4_http_client = adapter._build_sigv4_http_client()
            assert mock_build.call_count == 1

            # get_extra_client_params reuses the cached client, does NOT rebuild
            with patch.object(adapter, "get_request_provider_data", return_value=None):
                adapter.get_extra_client_params()
            assert mock_build.call_count == 1  # still 1, not 2


class TestBedrockInferenceAdapterAuthErrors:
    """Tests for user-facing auth error handling."""

    def test_sigv4_auth_error_preserves_detail_in_internal_server_error(self):
        """SigV4 auth failures should return a clear, generic 500 message."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig
        from ogx_api.common.errors import InternalServerError

        adapter = BedrockInferenceAdapter(config=BedrockConfig(region_name="us-east-1"))

        with pytest.raises(InternalServerError) as exc_info:
            adapter._handle_auth_error(
                "request signed with invalid credentials",
                RuntimeError("provider boom"),
                use_sigv4=True,
            )

        message = str(exc_info.value)
        assert (
            message
            == "Authentication failed because the configured cloud credentials could not authorize this request. "
            "Please verify that the credentials available to the server are valid, unexpired, and allowed to access the requested model."
        )
        assert "AWS_ROLE_ARN" not in message
        assert "Bedrock" not in message

    def test_bearer_auth_error_preserves_detail_in_internal_server_error(self):
        """Bearer auth failures should be actionable without exposing internal header/config details."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig
        from ogx_api.common.errors import InternalServerError

        adapter = BedrockInferenceAdapter(config=BedrockConfig(region_name="us-east-1"))

        with pytest.raises(InternalServerError) as exc_info:
            adapter._handle_auth_error(
                "Error code: 401 - invalid api key format",
                RuntimeError("provider boom"),
                use_sigv4=False,
            )

        message = str(exc_info.value)
        assert (
            message == "Authentication failed because the provided request credential was rejected. "
            "Please verify that the credential is valid, unexpired, and authorized for this request."
        )
        assert "x-ogx-provider-data" not in message
        assert "Bedrock" not in message

    def test_expired_bearer_auth_error_preserves_sanitized_detail(self):
        """Expired bearer auth failures should stay actionable without exposing config names."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig
        from ogx_api.common.errors import InternalServerError

        adapter = BedrockInferenceAdapter(config=BedrockConfig(region_name="us-east-1"))

        with pytest.raises(InternalServerError) as exc_info:
            adapter._handle_auth_error(
                "Bearer Token has expired",
                RuntimeError("provider boom"),
                use_sigv4=False,
            )

        message = str(exc_info.value)
        assert (
            message == "Authentication failed because the provided request credential has expired. "
            "Please refresh the credential and try again, or remove it so the server can use its configured cloud credentials."
        )
        assert "AWS_BEARER_TOKEN_BEDROCK" not in message
        assert "Bedrock" not in message


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestSigV4MockTransport:
    """Integration-style tests using httpx.MockTransport to verify SigV4 signing."""

    def test_sigv4_adds_aws4_signature_header(self):
        """SigV4 auth should add AWS4-HMAC-SHA256 Authorization header."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        # Track the request that gets sent
        captured_request = None

        def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = None

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-1", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                client.post(
                    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None
        auth_header = captured_request.headers.get("authorization", "")

        # Verify SigV4 signature format
        assert auth_header.startswith("AWS4-HMAC-SHA256"), f"Expected SigV4 header, got: {auth_header}"
        assert "Credential=" in auth_header
        assert "SignedHeaders=" in auth_header
        assert "Signature=" in auth_header

        # Verify NO Bearer token is present
        assert "Bearer" not in auth_header, "SigV4 auth should not contain Bearer token"

    def test_sigv4_no_bearer_header_when_empty_api_key(self):
        """When api_key is empty, no Bearer header should be added."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = None

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-1", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                client.post(
                    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None
        auth_header = captured_request.headers.get("authorization", "")

        # Authorization header should be SigV4, not Bearer
        assert "AWS4-HMAC-SHA256" in auth_header
        assert "Bearer" not in auth_header

    def test_sigv4_includes_security_token_for_sts(self):
        """SigV4 auth should include x-amz-security-token for STS credentials."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "ASIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = "FwoGZXIvYXdzEBYaDGTestSessionToken"

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-west-2", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                client.post(
                    "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None

        # Verify security token header is present for STS credentials
        assert "x-amz-security-token" in captured_request.headers
        assert captured_request.headers["x-amz-security-token"] == mock_frozen_creds.token

    def test_sigv4_replaces_existing_bearer_header(self):
        """SigV4 auth should replace any existing Bearer Authorization header."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = None

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-1", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                # Simulate what OpenAI SDK does: add Bearer <NOTUSED> header
                client.post(
                    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                    headers={"Authorization": "Bearer <NOTUSED>"},
                )

        assert captured_request is not None
        auth_header = captured_request.headers.get("authorization", "")

        # Verify SigV4 replaced the Bearer header (not appended)
        assert auth_header.startswith("AWS4-HMAC-SHA256"), f"Expected SigV4 header, got: {auth_header}"
        assert "Bearer" not in auth_header, "SigV4 auth should have replaced Bearer header"
        assert "<NOTUSED>" not in auth_header, "Placeholder should be removed"

    def test_sigv4_host_header_includes_port(self):
        """Host header should include port for non-default ports."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = None

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-1", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            # Use non-default port
            with httpx.Client(auth=auth, transport=transport) as client:
                client.post(
                    "https://localhost:8443/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None

        # Verify the Host header includes the port
        host_header = captured_request.headers.get("host", "")
        assert host_header == "localhost:8443", f"Expected host with port, got: {host_header}"

        # The signed Authorization header should include host in SignedHeaders
        auth_header = captured_request.headers.get("authorization", "")
        assert "host" in auth_header.lower()

        # Verify SigV4 signature format and no Bearer token
        assert auth_header.startswith("AWS4-HMAC-SHA256"), f"Expected SigV4 header, got: {auth_header}"
        assert "Bearer" not in auth_header, "SigV4 auth should not contain Bearer token"


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestWebIdentityFederation:
    """
    Tests for Web Identity Federation (IRSA, GitHub Actions OIDC).

    These tests verify that SigV4 auth works correctly with temporary credentials
    obtained via AssumeRoleWithWebIdentity, as used in:
    - Kubernetes/OpenShift with IRSA (IAM Roles for Service Accounts)
    - GitHub Actions with OIDC (aws-actions/configure-aws-credentials)
    """

    def test_web_identity_credentials_include_session_token(self):
        """
        Web identity credentials should include x-amz-security-token header.

        When using IRSA or GitHub Actions OIDC, boto3 calls AssumeRoleWithWebIdentity
        which returns temporary credentials with a session token. This token must
        be included in the x-amz-security-token header for the request to succeed.
        """
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        # Simulate credentials from AssumeRoleWithWebIdentity
        # Note: ASIA prefix indicates temporary credentials (vs AKIA for static)
        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "ASIAQWERTYUIOPASDFGH"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYzxcvbnm123"
        mock_frozen_creds.token = "IQoJb3JpZ2luX2VjEBYaCXVzLWVhc3QtMSJHMEUCIQDExample..."  # STS session token

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-2", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                client.post(
                    "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions",
                    json={
                        "model": "us.meta.llama3-2-1b-instruct-v1:0",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

        assert captured_request is not None

        # Verify STS session token is included
        assert "x-amz-security-token" in captured_request.headers
        assert captured_request.headers["x-amz-security-token"] == mock_frozen_creds.token

        # Verify SigV4 signature is present and valid format
        auth_header = captured_request.headers.get("authorization", "")
        assert auth_header.startswith("AWS4-HMAC-SHA256")
        assert "Credential=ASIAQWERTYUIOPASDFGH" in auth_header
        assert "bedrock/aws4_request" in auth_header

        # Verify no Bearer token (would conflict with SigV4)
        assert "Bearer" not in auth_header

    def test_adapter_uses_sigv4_with_web_identity_env(self, monkeypatch):
        """
        BedrockInferenceAdapter should use SigV4 when web identity env vars are set.

        This simulates the Kubernetes/GitHub Actions scenario where no bearer token
        is configured but AWS credentials are available via web identity federation.
        """
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        # Set web identity environment variables
        monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/test-role")
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/secrets/token")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")

        # Create adapter without bearer token (should trigger SigV4)
        config = BedrockConfig(region_name="us-east-2")
        adapter = BedrockInferenceAdapter(config=config)

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            # Should use SigV4 since no bearer token is configured
            assert adapter._should_use_sigv4() is True

            # API key should be placeholder to satisfy OpenAIMixin validation (OCI pattern)
            # SigV4 auth handler replaces Bearer header with proper SigV4 signature
            assert adapter.get_api_key() == "<NOTUSED>"

    def test_credential_refresh_returns_fresh_credentials(self):
        """
        SigV4 auth should get fresh credentials on each request.

        Web identity credentials are temporary and expire. boto3's credential
        chain handles refresh automatically, but we need to call get_frozen_credentials()
        on each request to get the current valid credentials.
        """
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        call_count = 0
        captured_requests = []

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(200, json={"status": "ok"})

        # Simulate credentials that change (as would happen after refresh)
        initial_creds = MagicMock()
        initial_creds.access_key = "ASIAFIRSTCREDENTIAL"
        initial_creds.secret_key = "firstSecretKey123"
        initial_creds.token = "firstSessionToken"

        refreshed_creds = MagicMock()
        refreshed_creds.access_key = "ASIASECONDCREDENTIAL"
        refreshed_creds.secret_key = "secondSecretKey456"
        refreshed_creds.token = "secondSessionToken"

        def get_frozen_credentials():
            nonlocal call_count
            call_count += 1
            # Return different credentials on second call (simulating refresh)
            return initial_creds if call_count == 1 else refreshed_creds

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_creds = MagicMock()
            mock_creds.get_frozen_credentials = get_frozen_credentials
            mock_session.get_credentials.return_value = mock_creds

            auth = BedrockSigV4Auth(region="us-east-2", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                # First request
                client.post(
                    "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )
                # Second request (after simulated credential refresh)
                client.post(
                    "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert len(captured_requests) == 2

        # First request should use initial credentials
        first_auth = captured_requests[0].headers.get("authorization", "")
        assert "ASIAFIRSTCREDENTIAL" in first_auth
        assert captured_requests[0].headers.get("x-amz-security-token") == "firstSessionToken"

        # Second request should use refreshed credentials
        second_auth = captured_requests[1].headers.get("authorization", "")
        assert "ASIASECONDCREDENTIAL" in second_auth
        assert captured_requests[1].headers.get("x-amz-security-token") == "secondSessionToken"


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestAsyncAuthFlow:
    """Tests for async auth flow to verify non-blocking behavior."""

    async def test_async_auth_flow_signs_request(self):
        """Async auth flow should sign requests without blocking the event loop."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        async def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = None

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-1", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            async with httpx.AsyncClient(auth=auth, transport=transport) as client:
                await client.post(
                    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None
        auth_header = captured_request.headers.get("authorization", "")

        # Verify SigV4 signature format
        assert auth_header.startswith("AWS4-HMAC-SHA256"), f"Expected SigV4 header, got: {auth_header}"
        assert "Credential=" in auth_header
        assert "SignedHeaders=" in auth_header
        assert "Signature=" in auth_header

        # Verify NO Bearer token is present
        assert "Bearer" not in auth_header, "SigV4 auth should not contain Bearer token"

    async def test_async_auth_flow_includes_session_token(self):
        """Async auth flow should include x-amz-security-token for STS credentials."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        captured_request = None

        async def capture_request(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, json={"status": "ok"})

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "ASIAQWERTYUIOPASDFGH"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYzxcvbnm123"
        mock_frozen_creds.token = "IQoJb3JpZ2luX2VjAsyncTest..."

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            auth = BedrockSigV4Auth(region="us-east-2", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            async with httpx.AsyncClient(auth=auth, transport=transport) as client:
                await client.post(
                    "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None

        # Verify STS session token is included
        assert "x-amz-security-token" in captured_request.headers
        assert captured_request.headers["x-amz-security-token"] == mock_frozen_creds.token
