# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.orcarouter.config import OrcaRouterImplConfig
from ogx.providers.remote.inference.orcarouter.orcarouter import OrcaRouterInferenceAdapter
from ogx_api import OpenAICompletionRequestWithExtraBody


class TestOrcaRouterConfig:
    """Tests for the OrcaRouter inference provider config and adapter wiring."""

    def test_default_base_url(self):
        config = OrcaRouterImplConfig(api_key="test-key")
        adapter = OrcaRouterInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == "https://api.orcarouter.ai/v1"

    def test_custom_base_url_from_config(self):
        custom_url = "https://custom.orcarouter.ai/v1"
        config = OrcaRouterImplConfig(api_key="test-key", base_url=custom_url)
        adapter = OrcaRouterInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == custom_url

    def test_sample_run_config_uses_env_placeholder(self):
        cfg = OrcaRouterImplConfig.sample_run_config()
        assert cfg["base_url"] == "https://api.orcarouter.ai/v1"
        assert cfg["api_key"] == "${env.ORCAROUTER_API_KEY:=}"

    def test_sample_run_config_env_expansion(self):
        config_data = OrcaRouterImplConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = OrcaRouterImplConfig.model_validate(processed_config)

        assert str(config.base_url) == "https://api.orcarouter.ai/v1"

    def test_provider_data_api_key_field(self):
        config = OrcaRouterImplConfig(api_key="test-key")
        adapter = OrcaRouterInferenceAdapter(config=config)
        assert adapter.provider_data_api_key_field == "orcarouter_api_key"

    async def test_legacy_completions_endpoint_not_supported(self):
        """OrcaRouter does not support the legacy /v1/completions endpoint."""
        config = OrcaRouterImplConfig(api_key="test-key")
        adapter = OrcaRouterInferenceAdapter(config=config)

        params = OpenAICompletionRequestWithExtraBody(model="anthropic/claude-sonnet-4.6", prompt="Hello")

        with pytest.raises(NotImplementedError, match="does not support /v1/completions endpoint"):
            await adapter.openai_completion(params)
