# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

import pytest

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.xai.config import XAIImplConfig
from ogx.providers.remote.inference.xai.xai import XAIInferenceAdapter
from ogx_api import OpenAICompletionRequestWithExtraBody


class TestXAIConfig:
    """Tests for the xAI (Grok) inference provider config and adapter wiring."""

    def test_default_base_url(self):
        config = XAIImplConfig(api_key="test-key")
        adapter = XAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None
        assert adapter.get_base_url() == "https://api.x.ai/v1"

    def test_custom_base_url_from_config(self):
        custom_url = "https://custom.x.ai/v1"
        config = XAIImplConfig(api_key="test-key", base_url=custom_url)
        adapter = XAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None
        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"XAI_BASE_URL": "https://env.x.ai/v1"})
    def test_base_url_from_environment_variable(self):
        config_data = XAIImplConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = XAIImplConfig.model_validate(processed_config)
        assert str(config.base_url) == "https://api.x.ai/v1"

    def test_sample_run_config_uses_env_placeholder(self):
        cfg = XAIImplConfig.sample_run_config()
        assert cfg["base_url"] == "https://api.x.ai/v1"
        assert cfg["api_key"] == "${env.XAI_API_KEY:=}"

    def test_provider_data_api_key_field(self):
        config = XAIImplConfig(api_key="test-key")
        adapter = XAIInferenceAdapter(config=config)
        assert adapter.provider_data_api_key_field == "xai_api_key"

    async def test_embeddings_not_supported(self):
        config = XAIImplConfig(api_key="test-key")
        adapter = XAIInferenceAdapter(config=config)
        with pytest.raises(NotImplementedError, match="does not expose an embeddings endpoint"):
            await adapter.openai_embeddings(None)  # type: ignore[arg-type]

    async def test_legacy_completions_endpoint_not_supported(self):
        """xAI does not support the legacy /v1/completions endpoint."""
        config = XAIImplConfig(api_key="test-key")
        adapter = XAIInferenceAdapter(config=config)

        params = OpenAICompletionRequestWithExtraBody(model="grok-4", prompt="Hello")

        with pytest.raises(NotImplementedError, match="does not support /v1/completions endpoint"):
            await adapter.openai_completion(params)
