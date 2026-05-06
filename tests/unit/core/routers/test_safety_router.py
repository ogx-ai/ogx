# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from ogx.core.datatypes import SafetyConfig
from ogx.core.routers.safety import SafetyRouter
from ogx_api import (
    ListShieldsResponse,
    ModerationObject,
    ModerationObjectResults,
    RunModerationRequest,
    Shield,
)


async def test_run_moderation_uses_default_shield_when_model_missing():
    routing_table = AsyncMock()
    shield = Shield(
        identifier="shield-1",
        provider_resource_id="provider/shield-model",
        provider_id="provider-id",
        params={},
    )
    routing_table.list_shields.return_value = ListShieldsResponse(data=[shield])

    moderation_response = ModerationObject(
        id="mid",
        model="shield-1",
        results=[ModerationObjectResults(flagged=False)],
    )
    provider = AsyncMock()
    provider.run_moderation.return_value = moderation_response
    routing_table.get_provider_impl.return_value = provider

    router = SafetyRouter(routing_table=routing_table, safety_config=SafetyConfig(default_shield_id="shield-1"))

    request = RunModerationRequest(input="hello world")
    result = await router.run_moderation(request)

    assert result is moderation_response
    routing_table.get_provider_impl.assert_awaited_once_with("shield-1")
    provider.run_moderation.assert_awaited_once()
    call_args = provider.run_moderation.call_args
    provider_request = call_args[0][0]
    assert provider_request.model == "provider/shield-model"
    assert provider_request.input == "hello world"
