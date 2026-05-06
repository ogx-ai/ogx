# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import ANY, MagicMock, patch

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.openai.config import OpenAIConfig
from ogx.providers.remote.inference.openai.openai import OpenAIInferenceAdapter


class TestOpenAIBaseURLConfig:
    """Test that OPENAI_BASE_URL environment variable properly configures the OpenAI adapter."""

    def test_default_base_url_without_env_var(self):
        """Test that the adapter uses the default OpenAI base URL when no environment variable is set."""
        config = OpenAIConfig(api_key="test-key")
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == "https://api.openai.com/v1"

    def test_custom_base_url_from_config(self):
        """Test that the adapter uses a custom base URL when provided in config."""
        custom_url = "https://custom.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.openai.com/v1"})
    def test_base_url_from_environment_variable(self):
        """Test that the adapter uses base URL from OPENAI_BASE_URL environment variable."""
        # Use sample_run_config which has proper environment variable syntax
        config_data = OpenAIConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = OpenAIConfig.model_validate(processed_config)
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        assert adapter.get_base_url() == "https://env.openai.com/v1"

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.openai.com/v1"})
    def test_config_overrides_environment_variable(self):
        """Test that explicit config value overrides environment variable."""
        custom_url = "https://config.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        # Config should take precedence over environment variable
        assert adapter.get_base_url() == custom_url

    @patch("ogx.providers.utils.inference.openai_mixin.AsyncOpenAI")
    def test_client_uses_configured_base_url(self, mock_openai_class):
        """Test that the OpenAI client is initialized with the configured base URL."""
        custom_url = "https://test.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        # Mock the get_api_key method since it's delegated to OpenAIMixin
        adapter.get_api_key = MagicMock(return_value="test-key")

        # Access the client property to trigger AsyncOpenAI initialization
        _ = adapter.client

        # Verify AsyncOpenAI was called with the correct base_url
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            base_url=custom_url,
            http_client=ANY,
        )

    @patch("ogx.providers.utils.inference.openai_mixin.AsyncOpenAI")
    async def test_check_model_availability_uses_configured_url(self, mock_openai_class):
        """Test that check_model_availability uses the configured base URL."""
        custom_url = "https://test.openai.com/v1"
        config = OpenAIConfig(api_key="test-key", base_url=custom_url)
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        # Mock the get_api_key method
        adapter.get_api_key = MagicMock(return_value="test-key")

        # Mock a model object that will be returned by models.list()
        mock_model = MagicMock()
        mock_model.id = "gpt-4"

        # Create an async iterator that yields our mock model
        async def mock_async_iterator():
            yield mock_model

        # Mock the AsyncOpenAI client and its models.list method
        mock_client = MagicMock()
        mock_client.models.list = MagicMock(return_value=mock_async_iterator())
        mock_openai_class.return_value = mock_client

        # Set the __provider_id__ attribute that's expected by list_models
        adapter.__provider_id__ = "openai"

        # Call check_model_availability and verify it returns True
        assert await adapter.check_model_availability("gpt-4")

        # Verify the client was created with the custom URL
        mock_openai_class.assert_called_with(
            api_key="test-key",
            base_url=custom_url,
            http_client=ANY,
        )

        # Verify the models.list method was called
        mock_client.models.list.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://proxy.openai.com/v1"})
    @patch("ogx.providers.utils.inference.openai_mixin.AsyncOpenAI")
    async def test_environment_variable_affects_model_availability_check(self, mock_openai_class):
        """Test that setting OPENAI_BASE_URL environment variable affects where model availability is checked."""
        # Use sample_run_config which has proper environment variable syntax
        config_data = OpenAIConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = OpenAIConfig.model_validate(processed_config)
        adapter = OpenAIInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None  # Disable provider data for this test

        # Mock the get_api_key method
        adapter.get_api_key = MagicMock(return_value="test-key")

        # Mock a model object that will be returned by models.list()
        mock_model = MagicMock()
        mock_model.id = "gpt-4"

        # Create an async iterator that yields our mock model
        async def mock_async_iterator():
            yield mock_model

        # Mock the AsyncOpenAI client and its models.list method
        mock_client = MagicMock()
        mock_client.models.list = MagicMock(return_value=mock_async_iterator())
        mock_openai_class.return_value = mock_client

        # Set the __provider_id__ attribute that's expected by list_models
        adapter.__provider_id__ = "openai"

        # Call check_model_availability and verify it returns True
        assert await adapter.check_model_availability("gpt-4")

        # Verify the client was created with the environment variable URL
        mock_openai_class.assert_called_with(
            api_key="test-key",
            base_url="https://proxy.openai.com/v1",
            http_client=ANY,
        )
