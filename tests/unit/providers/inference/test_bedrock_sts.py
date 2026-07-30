# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from ogx.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
from ogx.providers.remote.inference.bedrock.config import BedrockConfig
from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


def test_sigv4_auth_initialization():
    auth = BedrockSigV4Auth(
        region="us-east-1",
        aws_role_arn="arn:aws:iam::123:role/test",
        aws_web_identity_token_file="/tmp/token",
        aws_role_session_name="test-session",
        session_ttl=1800,
    )
    assert auth._region == "us-east-1"
    assert auth._aws_role_arn == "arn:aws:iam::123:role/test"
    assert auth._aws_web_identity_token_file == "/tmp/token"
    assert auth._aws_role_session_name == "test-session"
    assert auth._session_ttl == 1800


@pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
@patch("ogx.providers.utils.bedrock.sigv4_auth.logger")
def test_sigv4_auth_gets_refreshable_session(mock_logger):
    with patch("ogx.providers.utils.bedrock.refreshable_boto_session.RefreshableBotoSession") as mock_refreshable:
        mock_session = MagicMock()
        mock_refreshable.return_value.refreshable_session.return_value = mock_session

        auth = BedrockSigV4Auth(
            region="us-east-1",
            aws_role_arn="arn:aws:iam::123:role/test",
            aws_web_identity_token_file="/tmp/token",
        )

        auth._get_credentials()

        mock_refreshable.assert_called_once_with(
            region_name="us-east-1",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            profile_name=None,
            sts_arn="arn:aws:iam::123:role/test",
            web_identity_token_file="/tmp/token",
            session_name=None,
            session_ttl=3600,
        )
        assert auth._session == mock_session


def test_adapter_passes_sts_config_to_auth():
    config = BedrockConfig(
        region_name="us-west-2",
        aws_role_arn="arn:aws:iam::123:role/test",
        aws_web_identity_token_file="/tmp/token",
        session_ttl=1800,
    )
    adapter = BedrockInferenceAdapter(config=config)

    with patch("ogx.providers.utils.bedrock.sigv4_auth.BedrockSigV4Auth") as mock_auth:
        mock_auth.return_value = MagicMock()
        adapter._build_sigv4_http_client()

        mock_auth.assert_called_once_with(
            region="us-west-2",
            service="bedrock",
            aws_role_arn="arn:aws:iam::123:role/test",
            aws_web_identity_token_file="/tmp/token",
            session_ttl=1800,
        )
