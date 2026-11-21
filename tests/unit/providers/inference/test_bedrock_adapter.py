# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AuthenticationError
from pydantic import SecretStr

from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
from ogx.providers.remote.inference.bedrock.config import BedrockConfig
from ogx_api import InternalServerError, OpenAIChatCompletionRequestWithExtraBody


def test_adapter_initialization():
    config = BedrockConfig(aws_bedrock_bearer_token="test-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    assert adapter.config.auth_credential.get_secret_value() == "test-key"
    assert adapter.config.region_name == "us-east-1"


def test_client_url_construction():
    config = BedrockConfig(aws_bedrock_bearer_token="test-key", region_name="us-west-2")
    adapter = BedrockInferenceAdapter(config=config)

    assert adapter.get_base_url() == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"


def test_api_key_from_config():
    config = BedrockConfig(aws_bedrock_bearer_token="config-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)
    assert adapter.config.auth_credential.get_secret_value() == "config-key"


def test_api_key_from_header_overrides_config():
    """Test API key from request header overrides config via client property"""
    config = BedrockConfig(aws_bedrock_bearer_token="config-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)
    adapter.provider_data_api_key_field = "aws_bedrock_bearer_token"
    adapter.get_request_provider_data = MagicMock(
        return_value=SimpleNamespace(aws_bedrock_bearer_token=SecretStr("header-key"))
    )

    # The client property is where header override happens (in OpenAIMixin)
    assert adapter.client.api_key == "header-key"


async def test_authentication_error_handling():
    """Authentication failures should surface as a sanitized InternalServerError."""
    config = BedrockConfig(aws_bedrock_bearer_token="invalid-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    # Mock the parent class method to raise AuthenticationError
    mock_response = MagicMock()
    mock_response.message = "Invalid authentication credentials"
    auth_error = AuthenticationError(message="Invalid authentication credentials", response=mock_response, body=None)

    # Create a mock that raises the error
    mock_super = AsyncMock(side_effect=auth_error)

    # Patch the parent class method
    original_method = BedrockInferenceAdapter.__bases__[0].openai_chat_completion
    BedrockInferenceAdapter.__bases__[0].openai_chat_completion = mock_super

    try:
        with pytest.raises(InternalServerError) as exc_info:
            params = OpenAIChatCompletionRequestWithExtraBody(
                model="test-model", messages=[{"role": "user", "content": "test"}]
            )
            await adapter.openai_chat_completion(params=params)

        message = str(exc_info.value)
        assert (
            message == "Authentication failed because the provided request credential was rejected. "
            "Please verify that the credential is valid, unexpired, and authorized for this request."
        )
        assert "Bedrock" not in message
        assert "x-ogx-provider-data" not in message
    finally:
        # Restore original method
        BedrockInferenceAdapter.__bases__[0].openai_chat_completion = original_method


async def test_list_provider_model_ids_sigv4(monkeypatch):
    """Test dynamic model discovery includes both foundation models and inference profiles."""
    monkeypatch.delenv("AWS_BEDROCK_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("OGX_BEDROCK_BEARER_TOKEN", raising=False)
    config = BedrockConfig(aws_bedrock_bearer_token="", region_name="us-east-2")
    adapter = BedrockInferenceAdapter(config=config)

    mock_client = MagicMock()
    mock_client.list_foundation_models.return_value = {
        "modelSummaries": [
            {"modelId": "meta.llama3-1-8b-instruct-v1:0", "modelLifecycleStatus": "ACTIVE"},
            {"modelId": "deprecated-model", "modelLifecycleStatus": "LEGACY"},
        ]
    }
    mock_client.list_inference_profiles.return_value = {
        "inferenceProfileSummaries": [
            {"inferenceProfileId": "us.meta.llama3-1-8b-instruct-v1:0", "status": "ACTIVE"},
            {"inferenceProfileId": "us.meta.llama3-3-70b-instruct-v1:0", "status": "ACTIVE"},
            {"inferenceProfileId": "inactive-profile", "status": "INACTIVE"},
        ]
    }
    adapter._bedrock_client = mock_client

    models = await adapter.list_provider_model_ids()

    assert "meta.llama3-1-8b-instruct-v1:0" in models
    assert "us.meta.llama3-1-8b-instruct-v1:0" in models
    assert "us.meta.llama3-3-70b-instruct-v1:0" in models
    assert "deprecated-model" not in models
    assert "inactive-profile" not in models
    mock_client.list_foundation_models.assert_called_once_with(byInferenceType="ON_DEMAND")
    mock_client.list_inference_profiles.assert_called_once_with(typeEquals="SYSTEM_DEFINED")


async def test_list_provider_model_ids_sigv4_error_raises(monkeypatch):
    """Test that errors in list_foundation_models or list_inference_profiles re-raise."""
    monkeypatch.delenv("AWS_BEDROCK_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("OGX_BEDROCK_BEARER_TOKEN", raising=False)
    config = BedrockConfig(aws_bedrock_bearer_token="", region_name="us-east-2")
    adapter = BedrockInferenceAdapter(config=config)

    mock_client = MagicMock()
    mock_client.list_foundation_models.side_effect = RuntimeError("AWS Bedrock Error")
    adapter._bedrock_client = mock_client

    with pytest.raises(RuntimeError, match="AWS Bedrock Error"):
        await adapter.list_provider_model_ids()


async def test_list_provider_model_ids_bearer_token_error_raises(monkeypatch):
    """Test that errors in mantle endpoint model listing re-raise."""
    config = BedrockConfig(aws_bedrock_bearer_token="test-bearer-token", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    with patch.object(adapter, "get_api_key", return_value="test-bearer-token"):
        with patch("ogx.providers.remote.inference.bedrock.bedrock.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.models.list.side_effect = RuntimeError("Mantle Error")
            mock_openai.return_value = mock_client

            with pytest.raises(RuntimeError, match="Mantle Error"):
                await adapter.list_provider_model_ids()
