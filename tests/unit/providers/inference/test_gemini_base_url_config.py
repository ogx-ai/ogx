# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import ANY, MagicMock, patch

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.gemini.config import GeminiConfig
from ogx.providers.remote.inference.gemini.gemini import GeminiInferenceAdapter


class TestGeminiBaseURLConfig:
    """Test that GEMINI_BASE_URL environment variable properly configures the Gemini adapter."""

    def test_default_base_url_without_env_var(self):
        """Test that the adapter uses the default Gemini base URL when no environment variable is set."""
        config = GeminiConfig(api_key="test-key")
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == "https://generativelanguage.googleapis.com/v1beta/openai/"

    def test_custom_base_url_from_config(self):
        """Test that the adapter uses a custom base URL when provided in config."""
        custom_url = "https://custom.gemini-proxy.com/v1"
        config = GeminiConfig(api_key="test-key", base_url=custom_url)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"GEMINI_BASE_URL": "https://env.gemini-proxy.com/v1"})
    def test_base_url_from_environment_variable(self):
        """Test that the adapter uses base URL from GEMINI_BASE_URL environment variable."""
        config_data = GeminiConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = GeminiConfig.model_validate(processed_config)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == "https://env.gemini-proxy.com/v1"

    @patch.dict(os.environ, {"GEMINI_BASE_URL": "https://env.gemini-proxy.com/v1"})
    def test_config_overrides_environment_variable(self):
        """Test that explicit config value overrides environment variable."""
        custom_url = "https://config.gemini-proxy.com/v1"
        config = GeminiConfig(api_key="test-key", base_url=custom_url)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == custom_url

    @patch("ogx.providers.utils.inference.openai_mixin.AsyncOpenAI")
    def test_client_uses_configured_base_url(self, mock_openai_class):
        """Test that the client is initialized with the configured base URL."""
        custom_url = "https://test.gemini-proxy.com/v1"
        config = GeminiConfig(api_key="test-key", base_url=custom_url)
        adapter = GeminiInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        adapter.get_api_key = MagicMock(return_value="test-key")

        # Access the client property to trigger AsyncOpenAI initialization
        _ = adapter.client

        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url=custom_url,
            http_client=ANY,
        )
