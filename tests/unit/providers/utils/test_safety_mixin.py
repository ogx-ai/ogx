# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from ogx.providers.inline.safety.prompt_guard.prompt_guard import (
    PromptGuardSafetyImpl,
)
from ogx.providers.remote.safety.bedrock.bedrock import BedrockSafetyAdapter
from ogx.providers.remote.safety.nvidia.nvidia import NVIDIASafetyAdapter
from ogx.providers.remote.safety.sambanova.sambanova import (
    SambaNovaSafetyAdapter,
)
from ogx.providers.utils.safety import ShieldToModerationMixin
from ogx_api import (
    OpenAIUserMessageParam,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ShieldsProtocolPrivate,
    ViolationLevel,
)


@pytest.mark.parametrize(
    "provider_class",
    [
        NVIDIASafetyAdapter,
        BedrockSafetyAdapter,
        SambaNovaSafetyAdapter,
        PromptGuardSafetyImpl,
    ],
)
def test_providers_use_mixin(provider_class):
    """Providers should use mixin for run_moderation ."""
    for cls in provider_class.__mro__:
        if "run_moderation" in cls.__dict__:
            assert cls.__name__ == "ShieldToModerationMixin"
            return
    pytest.fail(f"{provider_class.__name__} has no run_moderation")


async def test_safe_content():
    """Safe content returns unflagged result."""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        def __init__(self):
            self.run_shield = AsyncMock(return_value=RunShieldResponse(violation=None))

    request = RunModerationRequest(input="safe", model="test")
    result = await Provider().run_moderation(request)

    assert result is not None
    assert result.results[0].flagged is False


async def test_unsafe_content():
    """Unsafe content returns flagged with violation_type as category."""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        def __init__(self):
            self.run_shield = AsyncMock(
                return_value=RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message="Blocked",
                        metadata={"violation_type": "harmful"},
                    )
                )
            )

    request = RunModerationRequest(input="unsafe", model="test")
    result = await Provider().run_moderation(request)

    assert result.results[0].flagged is True
    assert result.results[0].categories == {"harmful": True}
    assert result.results[0].category_scores == {"harmful": 1.0}


async def test_missing_violation_type_defaults_to_unsafe():
    """When violation_type missing in metadata, defaults to 'unsafe'."""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        def __init__(self):
            self.run_shield = AsyncMock(
                return_value=RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message="Bad",
                        metadata={},  # No violation_type
                    )
                )
            )

    request = RunModerationRequest(input="test", model="test")
    result = await Provider().run_moderation(request)

    assert result.results[0].categories == {"unsafe": True}


async def test_non_string_violation_type_defaults_to_unsafe():
    """Non-string violation_type defaults to 'unsafe'"""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        def __init__(self):
            self.run_shield = AsyncMock(
                return_value=RunShieldResponse(
                    violation=SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message="Bad",
                        metadata={"violation_type": 12345},  # int, not string
                    )
                )
            )

    request = RunModerationRequest(input="test", model="test")
    result = await Provider().run_moderation(request)

    assert result.results[0].categories == {"unsafe": True}


async def test_multiple_inputs():
    """List input produces multiple results."""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        def __init__(self):
            self.run_shield = AsyncMock()

    provider = Provider()
    provider.run_shield.side_effect = [
        RunShieldResponse(violation=None),
        RunShieldResponse(
            violation=SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message="Bad",
                metadata={"violation_type": "bad"},
            )
        ),
    ]

    request = RunModerationRequest(input=["safe", "unsafe"], model="test")
    result = await provider.run_moderation(request)

    assert len(result.results) == 2
    assert result.results[0].flagged is False
    assert result.results[1].flagged is True


async def test_run_shield_receives_correct_params():
    """Verify run_shield called with RunShieldRequest containing shield_id and messages."""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        def __init__(self):
            self.run_shield = AsyncMock(return_value=RunShieldResponse(violation=None))

    provider = Provider()
    request = RunModerationRequest(input="test input", model="my-shield")
    await provider.run_moderation(request)

    call_args = provider.run_shield.call_args.args
    assert len(call_args) == 1
    shield_request = call_args[0]
    assert isinstance(shield_request, RunShieldRequest)
    assert shield_request.shield_id == "my-shield"
    assert isinstance(shield_request.messages[0], OpenAIUserMessageParam)
    assert shield_request.messages[0].content == "test input"


async def test_model_none_raises_error():
    """Model parameter is required (cannot be None)."""

    class Provider(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
        pass

    request = RunModerationRequest(input="test", model=None)
    with pytest.raises(ValueError, match="moderation requires a model identifier"):
        await Provider().run_moderation(request)
