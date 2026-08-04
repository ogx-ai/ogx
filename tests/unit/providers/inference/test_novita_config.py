# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.novita.config import NovitaImplConfig
from ogx.providers.remote.inference.novita.novita import NovitaInferenceAdapter


class TestNovitaConfig:
    """Tests for the Novita inference provider config and adapter wiring."""

    def test_default_base_url(self):
        config = NovitaImplConfig(api_key="test-key")
        adapter = NovitaInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == "https://api.novita.ai/openai"

    def test_custom_base_url_from_config(self):
        custom_url = "https://custom.novita.ai/openai"
        config = NovitaImplConfig(api_key="test-key", base_url=custom_url)
        adapter = NovitaInferenceAdapter(config=config)
        adapter.provider_data_api_key_field = None

        assert adapter.get_base_url() == custom_url

    @patch.dict(os.environ, {"NOVITA_BASE_URL": "https://env.novita.ai/openai"})
    def test_base_url_from_environment_variable(self):
        config_data = NovitaImplConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = NovitaImplConfig.model_validate(processed_config)

        assert str(config.base_url) == "https://api.novita.ai/openai"

    def test_sample_run_config_uses_env_placeholder(self):
        cfg = NovitaImplConfig.sample_run_config()
        assert cfg["base_url"] == "https://api.novita.ai/openai"
        assert cfg["api_key"] == "${env.NOVITA_API_KEY:=}"

    def test_provider_data_api_key_field(self):
        config = NovitaImplConfig(api_key="test-key")
        adapter = NovitaInferenceAdapter(config=config)
        assert adapter.provider_data_api_key_field == "novita_api_key"
