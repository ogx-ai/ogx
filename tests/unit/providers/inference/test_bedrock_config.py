# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.remote.inference.bedrock.config import BedrockConfig


def test_bedrock_config_defaults_no_env(monkeypatch):
    """Test BedrockConfig defaults when env vars are not set"""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    config = BedrockConfig()
    assert config.auth_credential is None
    assert config.region_name == "us-east-2"


def test_bedrock_config_reads_from_env(monkeypatch):
    """Test BedrockConfig field initialization reads from environment variables"""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    config = BedrockConfig()
    assert config.region_name == "eu-west-1"


def test_bedrock_config_with_values():
    """Test BedrockConfig accepts explicit values via the canonical field name."""
    config = BedrockConfig(aws_bedrock_bearer_token="test-key", region_name="us-west-2")
    assert config.auth_credential.get_secret_value() == "test-key"
    assert config.region_name == "us-west-2"


def test_bedrock_config_legacy_api_key_alias_still_works():
    """Test BedrockConfig keeps accepting the legacy api_key alias."""
    config = BedrockConfig(api_key="legacy-key", region_name="us-west-2")
    assert config.auth_credential.get_secret_value() == "legacy-key"


def test_bedrock_config_sample():
    """Test BedrockConfig sample_run_config returns correct format"""
    sample = BedrockConfig.sample_run_config()
    assert "aws_bedrock_bearer_token" in sample
    assert "region_name" in sample
    assert "aws_role_arn" in sample
    assert "aws_web_identity_token_file" in sample
    assert sample["aws_bedrock_bearer_token"] == "${env.AWS_BEDROCK_BEARER_TOKEN:=}"
    assert sample["region_name"] == "${env.AWS_DEFAULT_REGION:=us-east-2}"
    assert sample["aws_role_arn"] == "${env.AWS_ROLE_ARN:=}"
    assert sample["aws_web_identity_token_file"] == "${env.AWS_WEB_IDENTITY_TOKEN_FILE:=}"


def test_bedrock_config_sts_fields(monkeypatch):
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123:role/test")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/tmp/token")
    config = BedrockConfig()
    assert config.aws_role_arn == "arn:aws:iam::123:role/test"
    assert config.aws_web_identity_token_file == "/tmp/token"
