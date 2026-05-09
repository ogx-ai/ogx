# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

from ogx.providers.utils.bedrock.config import BedrockBaseConfig


class TestBedrockBaseConfig:
    def test_defaults_work_without_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            config = BedrockBaseConfig()

            # Basic creds should be None
            assert config.aws_access_key_id is None
            assert config.aws_secret_access_key is None
            assert config.region_name is None

            # Timeouts get defaults
            assert config.connect_timeout == 60.0
            assert config.read_timeout == 60.0
            assert config.session_ttl == 3600

    def test_env_vars_get_picked_up(self):
        env_vars = {
            "AWS_ACCESS_KEY_ID": "AKIATEST123",
            "AWS_SECRET_ACCESS_KEY": "secret123",
            "AWS_DEFAULT_REGION": "us-west-2",
            "AWS_MAX_ATTEMPTS": "5",
            "AWS_RETRY_MODE": "adaptive",
            "AWS_CONNECT_TIMEOUT": "30",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = BedrockBaseConfig()

            assert config.aws_access_key_id.get_secret_value() == "AKIATEST123"
            assert config.aws_secret_access_key.get_secret_value() == "secret123"
            assert config.region_name == "us-west-2"
            assert config.total_max_attempts == 5
            assert config.retry_mode == "adaptive"
            assert config.connect_timeout == 30.0

    def test_partial_env_setup(self):
        # Just setting one timeout var
        with patch.dict(os.environ, {"AWS_CONNECT_TIMEOUT": "120"}, clear=True):
            config = BedrockBaseConfig()

            assert config.connect_timeout == 120.0
            assert config.read_timeout == 60.0  # still default
            assert config.aws_access_key_id is None

    def test_bad_max_attempts_breaks(self):
        with patch.dict(os.environ, {"AWS_MAX_ATTEMPTS": "not_a_number"}, clear=True):
            try:
                BedrockBaseConfig()
                raise AssertionError("Should have failed on bad int conversion")
            except ValueError:
                pass  # expected
