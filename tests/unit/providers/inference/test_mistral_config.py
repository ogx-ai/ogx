# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import AsyncMock, patch

import pytest

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.mistral.config import MistralImplConfig
from ogx.providers.remote.inference.mistral.mistral import MistralInferenceAdapter
from ogx_api import (
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
)


class TestMistralConfig:
    """Tests for the Mistral inference provider config and adapter wiring."""

    def test_default_base_url(self):
        config = MistralImplConfig(api_key="test-key")
        adapter = MistralInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == "https://api.mistral.ai/v1"

    def test_custom_base_url_from_config(self):
        custom_url = "https://custom.mistral.ai/v1"
        config = MistralImplConfig(api_key="test-key", base_url=custom_url)
        adapter = MistralInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"MISTRAL_BASE_URL": "https://env.mistral.ai/v1"})
    def test_base_url_from_environment_variable(self):
        config_data = MistralImplConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = MistralImplConfig.model_validate(processed_config)

        assert str(config.base_url) == "https://api.mistral.ai/v1"

    def test_sample_run_config_uses_env_placeholder(self):
        cfg = MistralImplConfig.sample_run_config()
        assert cfg["base_url"] == "https://api.mistral.ai/v1"
        assert cfg["api_key"] == "${env.MISTRAL_API_KEY:=}"

    def test_provider_data_api_key_field(self):
        config = MistralImplConfig(api_key="test-key")
        adapter = MistralInferenceAdapter(config=config)
        assert adapter.provider_data_api_key_field == "mistral_api_key"
        assert "mistral-embed" in adapter.embedding_model_metadata

    async def test_base64_encoding_format_raises_value_error(self):
        """Mistral's embeddings endpoint does not support encoding_format='base64'."""
        config = MistralImplConfig(api_key="test-key", base_url="https://api.mistral.ai/v1")
        adapter = MistralInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None
        adapter.model_store = AsyncMock()
        adapter.model_store.has_model = AsyncMock(return_value=True)
        adapter.model_store.get_model = AsyncMock()

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model="mistral-embed",
            input="hello world",
            encoding_format="base64",
        )

        with pytest.raises(ValueError, match="does not support encoding_format='base64'"):
            await adapter.openai_embeddings(params)

    async def test_legacy_completions_endpoint_not_supported(self):
        """Mistral does not support the legacy /v1/completions endpoint."""
        config = MistralImplConfig(api_key="test-key", base_url="https://api.mistral.ai/v1")
        adapter = MistralInferenceAdapter(config=config)

        params = OpenAICompletionRequestWithExtraBody(model="mistral-small", prompt="Hello")

        with pytest.raises(NotImplementedError, match="does not support /v1/completions endpoint"):
            await adapter.openai_completion(params)
