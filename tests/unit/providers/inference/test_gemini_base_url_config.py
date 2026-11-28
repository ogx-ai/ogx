# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import ANY, MagicMock, patch

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.gemini.config import DEFAULT_BASE_URL, GeminiConfig
from ogx.providers.remote.inference.gemini.gemini import GeminiInferenceAdapter


class TestGeminiBaseURLConfig:
    """Test that GEMINI_BASE_URL environment variable and base_url config properly configure the Gemini adapter (#7041)."""

    def test_default_base_url_without_env_var(self):
        """Test that the adapter uses the default Gemini base URL when no custom URL is provided."""
        config = GeminiConfig(api_key="test-key")
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == DEFAULT_BASE_URL
        assert adapter.get_base_url() == "https://generativelanguage.googleapis.com/v1beta/openai/"

    def test_custom_base_url_from_config(self):
        """Test that the adapter uses a custom base URL when provided in config."""
        custom_url = "http://localhost:9999/v1"
        config = GeminiConfig(api_key="test-key", base_url=custom_url)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == custom_url

    def test_sample_run_config_templates_base_url(self):
        """sample_run_config must emit an env-templated default for base_url."""
        config_data = GeminiConfig.sample_run_config(api_key="test-key")
        assert config_data["base_url"] == f"${{env.GEMINI_BASE_URL:={DEFAULT_BASE_URL}}}"

    @patch.dict(os.environ, {"GEMINI_BASE_URL": "https://gateway.internal.net/gemini/v1"})
    def test_base_url_from_environment_variable(self):
        """Test that the adapter uses base URL from GEMINI_BASE_URL environment variable."""
        config_data = GeminiConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = GeminiConfig.model_validate(processed_config)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == "https://gateway.internal.net/gemini/v1"

    @patch.dict(os.environ, {"GEMINI_BASE_URL": "https://env.gemini.com/v1"})
    def test_config_overrides_environment_variable(self):
        """Test that explicit config value overrides environment variable."""
        custom_url = "https://config.gemini.com/v1"
        config = GeminiConfig(api_key="test-key", base_url=custom_url)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == custom_url

    @patch("ogx.providers.utils.inference.openai_mixin.AsyncOpenAI")
    def test_client_uses_configured_base_url(self, mock_openai_class):
        """Test that the OpenAI client is initialized with the configured base URL."""
        custom_url = "http://localhost:9999/v1"
        config = GeminiConfig(api_key="test-key", base_url=custom_url)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        adapter.get_api_key = MagicMock(return_value="test-key")

        _ = adapter.client

        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url=custom_url,
            http_client=ANY,
        )
