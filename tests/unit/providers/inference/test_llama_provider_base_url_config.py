# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

from ogx.core.stack import replace_env_vars
from ogx.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from ogx.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig


class TestLlamaCompatBaseURLConfig:
    """LLAMA_API_BASE_URL must actually reach the generated config (#6604)."""

    def test_sample_run_config_templates_base_url(self):
        """sample_run_config must emit an env-templated default, not a hardcoded URL,
        so generated configs pick up LLAMA_API_BASE_URL like every other remote provider."""
        config_data = LlamaCompatConfig.sample_run_config(api_key="test-key")
        assert config_data["base_url"] == "${env.LLAMA_API_BASE_URL:=https://api.llama.com/compat/v1/}"

    def test_default_base_url_without_env_var(self):
        """With no env var set, replace_env_vars resolves to the original default."""
        config_data = LlamaCompatConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = LlamaCompatConfig.model_validate(processed_config)

        assert str(config.base_url) == "https://api.llama.com/compat/v1/"

    @patch.dict(os.environ, {"LLAMA_API_BASE_URL": "http://localhost:9090/v1"})
    def test_base_url_from_environment_variable(self):
        """Setting LLAMA_API_BASE_URL must override the default in the generated config."""
        config_data = LlamaCompatConfig.sample_run_config(api_key="test-key")
        processed_config = replace_env_vars(config_data)
        config = LlamaCompatConfig.model_validate(processed_config)

        assert str(config.base_url) == "http://localhost:9090/v1"


class TestLlamaCppServerBaseURLConfig:
    """LLAMA_CPP_SERVER_URL must actually reach the generated config (#6604, regression guard).

    sample_run_config already templated this correctly; these tests just lock that in since
    the sibling llama-openai-compat provider regressed on the identical pattern.
    """

    def test_sample_run_config_templates_base_url(self):
        config_data = LlamaCppServerConfig.sample_run_config()
        assert config_data["base_url"] == "${env.LLAMA_CPP_SERVER_URL:=http://localhost:8080/v1}"

    def test_default_base_url_without_env_var(self):
        config_data = LlamaCppServerConfig.sample_run_config()
        processed_config = replace_env_vars(config_data)
        config = LlamaCppServerConfig.model_validate(processed_config)

        assert str(config.base_url) == "http://localhost:8080/v1"

    @patch.dict(os.environ, {"LLAMA_CPP_SERVER_URL": "http://localhost:9090/v1"})
    def test_base_url_from_environment_variable(self):
        config_data = LlamaCppServerConfig.sample_run_config()
        processed_config = replace_env_vars(config_data)
        config = LlamaCppServerConfig.model_validate(processed_config)

        assert str(config.base_url) == "http://localhost:9090/v1"
