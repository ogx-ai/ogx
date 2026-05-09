# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx.providers.remote.safety.passthrough.config import (
    PassthroughProviderDataValidator,
    PassthroughSafetyConfig,
)


def test_config_requires_base_url():
    with pytest.raises(ValidationError):
        PassthroughSafetyConfig()  # type: ignore[call-arg]


def test_config_api_key_stored_as_secret():
    cfg = PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        api_key="secret-key",
    )
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "secret-key"


def test_config_forward_headers_accepts_mapping():
    cfg = PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        forward_headers={"maas_api_token": "Authorization"},
    )
    assert cfg.forward_headers == {"maas_api_token": "Authorization"}


def test_config_extra_blocked_headers_accepts_list():
    cfg = PassthroughSafetyConfig(
        base_url="https://safety.example.com/v1",
        extra_blocked_headers=["x-internal-debug"],
    )
    assert cfg.extra_blocked_headers == ["x-internal-debug"]


def test_config_sample_run_config():
    sample = PassthroughSafetyConfig.sample_run_config()
    assert "base_url" in sample
    assert "api_key" in sample


def test_config_sample_run_config_includes_extra_blocked_headers_when_set():
    sample = PassthroughSafetyConfig.sample_run_config(extra_blocked_headers=["x-internal-debug"])
    assert sample["extra_blocked_headers"] == ["x-internal-debug"]


def test_provider_data_validator_allows_extra_keys():
    v = PassthroughProviderDataValidator(passthrough_api_key="my-key", custom_field="val")
    assert v.passthrough_api_key is not None
    assert v.passthrough_api_key.get_secret_value() == "my-key"
