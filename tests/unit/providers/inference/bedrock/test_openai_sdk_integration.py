# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for Bedrock OpenAI SDK integration with SigV4 authentication.

These tests verify:
1. Base URL uses bedrock-runtime hostname
2. SigV4 signing uses "bedrock" as the service name (NOT "bedrock-runtime")
3. In SigV4 mode, no Bearer Authorization header is present
4. STS credentials work properly with temporary tokens
"""

import importlib.util
from unittest.mock import MagicMock, patch

import httpx
import pytest

HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
class TestBedrockOpenAISDKIntegration:
    """Tests for Bedrock OpenAI SDK integration with SigV4 auth."""

    def test_base_url_uses_bedrock_runtime_hostname(self):
        """Base URL should use bedrock-runtime hostname (endpoint prefix)."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        base_url = adapter.get_base_url()
        # Hostname uses "bedrock-runtime" (endpoint prefix)
        assert base_url == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"

    def test_sigv4_uses_bedrock_signing_name_not_bedrock_runtime(self):
        """
        SigV4 signing must use 'bedrock' as the service name, NOT 'bedrock-runtime'.

        The hostname is bedrock-runtime.<region>.amazonaws.com (endpoint prefix),
        but the SigV4 credential scope uses the signing name 'bedrock'.
        This is defined in botocore's service metadata.
        """
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-west-2")
        adapter = BedrockInferenceAdapter(config=config)

        with patch("ogx.providers.utils.bedrock.sigv4_auth.BedrockSigV4Auth") as mock_auth_cls:
            mock_auth_cls.return_value = MagicMock()
            adapter._build_sigv4_http_client()

            # Verify signing name is "bedrock", NOT "bedrock-runtime"
            call_kwargs = mock_auth_cls.call_args[1]
            assert call_kwargs["service"] == "bedrock", (
                "SigV4 must use signing name 'bedrock', not endpoint prefix 'bedrock-runtime'"
            )

    def test_sigv4_mode_uses_placeholder_api_key(self):
        """In SigV4 mode, api_key should be a placeholder (SigV4 auth replaces the header)."""
        from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        config = BedrockConfig(region_name="us-east-1")
        adapter = BedrockInferenceAdapter(config=config)

        with patch.object(adapter, "get_request_provider_data", return_value=None):
            # Patch SigV4Auth to avoid actual boto3 calls
            with patch("ogx.providers.utils.bedrock.sigv4_auth.BedrockSigV4Auth") as mock_auth_cls:
                mock_auth = MagicMock()
                mock_auth_cls.return_value = mock_auth

                client = adapter.client

                # OpenAI SDK requires a non-empty api_key for validation.
                # We use a placeholder that SigV4 auth replaces with proper signature.
                # This follows the same pattern as the OCI provider.
                assert client.api_key == "<NOTUSED>"

    def test_sigv4_authorization_header_format(self):
        """SigV4 Authorization header should start with AWS4-HMAC-SHA256, not Bearer."""
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

            # Use "bedrock" signing name (correct)
            auth = BedrockSigV4Auth(region="us-west-2", service="bedrock")
            transport = httpx.MockTransport(capture_request)

            with httpx.Client(auth=auth, transport=transport) as client:
                client.post(
                    "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
                    json={"model": "test"},
                )

        assert captured_request is not None
        auth_header = captured_request.headers.get("authorization", "")

        # Must be SigV4, NOT Bearer
        assert auth_header.startswith("AWS4-HMAC-SHA256"), f"Expected SigV4 header, got: {auth_header}"
        assert "Bearer" not in auth_header, "SigV4 auth should not contain Bearer"

    def test_sts_credentials_include_security_token(self):
        """SigV4 auth should include x-amz-security-token for STS credentials."""
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        mock_frozen_creds = MagicMock()
        mock_frozen_creds.access_key = "ASIAIOSFODNN7EXAMPLE"
        mock_frozen_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_frozen_creds.token = "AQoDYXdzEJr...<remainder of security token>"

        with patch("boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            mock_session.get_credentials.return_value.get_frozen_credentials.return_value = mock_frozen_creds

            # Use "bedrock" signing name (correct)
            auth = BedrockSigV4Auth(region="us-west-2", service="bedrock")

            request = httpx.Request(
                method="POST",
                url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
                headers={"content-type": "application/json"},
                content=b'{"model": "test"}',
            )

            gen = auth.auth_flow(request)
            signed_request = next(gen)

            # Verify session token header is present for STS credentials
            assert "x-amz-security-token" in signed_request.headers
            assert signed_request.headers["x-amz-security-token"] == mock_frozen_creds.token
