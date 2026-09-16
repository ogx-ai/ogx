# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

import pytest

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.thegrid.config import TheGridImplConfig
from ogx.providers.remote.inference.thegrid.thegrid import TheGridInferenceAdapter
from ogx_api import OpenAICompletionRequestWithExtraBody


class TestTheGridConfig:
    """Tests for The Grid inference provider config and adapter wiring."""

    def test_default_base_url(self):
        config = TheGridImplConfig(api_key="test-key")
        adapter = TheGridInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == "https://api.thegrid.ai/v1"

    def test_custom_base_url_from_config(self):
        custom_url = "https://custom.thegrid.ai/v1"
        config = TheGridImplConfig(api_key="test-key", base_url=custom_url)
        adapter = TheGridInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"THEGRID_BASE_URL": "https://env.thegrid.ai/v1"})
    def test_base_url_from_environment_variable(self):
        config_data = TheGridImplConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = TheGridImplConfig.model_validate(processed_config)

        assert str(config.base_url) == "https://api.thegrid.ai/v1"

    def test_sample_run_config_uses_env_placeholder(self):
        cfg = TheGridImplConfig.sample_run_config()
        assert cfg["base_url"] == "https://api.thegrid.ai/v1"
        assert cfg["api_key"] == "${env.THEGRID_API_KEY:=}"

    def test_provider_data_api_key_field(self):
        config = TheGridImplConfig(api_key="test-key")
        adapter = TheGridInferenceAdapter(config=config)
        assert adapter.provider_data_api_key_field == "thegrid_api_key"

    async def test_embeddings_not_supported(self):
        config = TheGridImplConfig(api_key="test-key")
        adapter = TheGridInferenceAdapter(config=config)
        with pytest.raises(NotImplementedError, match="does not expose an embeddings endpoint"):
            await adapter.openai_embeddings(None)  # type: ignore[arg-type]

    async def test_legacy_completions_endpoint_not_supported(self):
        """The Grid serves only /v1/chat/completions; the legacy route returns 404."""
        config = TheGridImplConfig(api_key="test-key")
        adapter = TheGridInferenceAdapter(config=config)

        params = OpenAICompletionRequestWithExtraBody(model="text-standard", prompt="Hello")

        with pytest.raises(NotImplementedError, match="does not support /v1/completions endpoint"):
            await adapter.openai_completion(params)
