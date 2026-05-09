# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core.datatypes import CORSConfig, process_cors_config


def test_cors_config_defaults():
    config = CORSConfig()

    assert config.allow_origins == []
    assert config.allow_origin_regex is None
    assert config.allow_methods == ["OPTIONS"]
    assert config.allow_headers == []
    assert config.allow_credentials is False
    assert config.expose_headers == []
    assert config.max_age == 600


def test_cors_config_explicit_config():
    config = CORSConfig(
        allow_origins=["https://example.com"], allow_credentials=True, max_age=3600, allow_methods=["GET", "POST"]
    )

    assert config.allow_origins == ["https://example.com"]
    assert config.allow_credentials is True
    assert config.max_age == 3600
    assert config.allow_methods == ["GET", "POST"]


def test_cors_config_regex():
    config = CORSConfig(allow_origins=[], allow_origin_regex=r"https?://localhost:\d+")

    assert config.allow_origins == []
    assert config.allow_origin_regex == r"https?://localhost:\d+"


def test_cors_config_wildcard_credentials_error():
    with pytest.raises(ValueError, match="Cannot use wildcard origins with credentials enabled"):
        CORSConfig(allow_origins=["*"], allow_credentials=True)

    with pytest.raises(ValueError, match="Cannot use wildcard origins with credentials enabled"):
        CORSConfig(allow_origins=["https://example.com", "*"], allow_credentials=True)


def test_process_cors_config_false():
    result = process_cors_config(False)
    assert result is None


def test_process_cors_config_true():
    result = process_cors_config(True)

    assert isinstance(result, CORSConfig)
    assert result.allow_origins == []
    assert result.allow_origin_regex == r"https?://localhost:\d+"
    assert result.allow_credentials is False
    expected_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    for method in expected_methods:
        assert method in result.allow_methods


def test_process_cors_config_passthrough():
    original = CORSConfig(allow_origins=["https://example.com"], allow_methods=["GET"])
    result = process_cors_config(original)

    assert result is original


def test_process_cors_config_invalid_type():
    with pytest.raises(ValueError, match="Expected bool or CORSConfig, got str"):
        process_cors_config("invalid")


def test_cors_config_model_dump():
    cors_config = CORSConfig(
        allow_origins=["https://example.com"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=True,
        max_age=3600,
    )

    config_dict = cors_config.model_dump()

    assert config_dict["allow_origins"] == ["https://example.com"]
    assert config_dict["allow_methods"] == ["GET", "POST"]
    assert config_dict["allow_headers"] == ["Content-Type"]
    assert config_dict["allow_credentials"] is True
    assert config_dict["max_age"] == 3600

    expected_keys = {
        "allow_origins",
        "allow_origin_regex",
        "allow_methods",
        "allow_headers",
        "allow_credentials",
        "expose_headers",
        "max_age",
    }
    assert set(config_dict.keys()) == expected_keys
