# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, call, patch

from ogx.providers.remote.safety.bedrock.bedrock import BedrockSafetyAdapter
from ogx.providers.remote.safety.bedrock.config import BedrockSafetyConfig


async def test_bedrock_safety_initialize_creates_clients():
    config = BedrockSafetyConfig(
        region_name="us-west-2",
        aws_role_arn="arn:aws:iam::123:role/test",
        aws_web_identity_token_file="/tmp/token",
    )
    adapter = BedrockSafetyAdapter(config=config)

    runtime_client = MagicMock(name="bedrock-runtime-client")
    bedrock_client = MagicMock(name="bedrock-client")
    with patch("ogx.providers.remote.safety.bedrock.bedrock.create_bedrock_client") as mock_create:
        mock_create.side_effect = [runtime_client, bedrock_client]

        await adapter.initialize()

        assert adapter.bedrock_runtime_client is runtime_client
        assert adapter.bedrock_client is bedrock_client
        mock_create.assert_has_calls(
            [
                call(config),
                call(config, "bedrock"),
            ]
        )
