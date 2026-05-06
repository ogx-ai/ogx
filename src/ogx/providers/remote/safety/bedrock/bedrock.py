# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from ogx.log import get_logger
from ogx.providers.utils.bedrock.client import create_bedrock_client
from ogx.providers.utils.safety import ShieldToModerationMixin
from ogx_api import (
    GetShieldRequest,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    Shield,
    ShieldsProtocolPrivate,
    ViolationLevel,
)

from .config import BedrockSafetyConfig

logger = get_logger(name=__name__, category="safety::bedrock")


class BedrockSafetyAdapter(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
    """Safety adapter for content moderation using AWS Bedrock Guardrails."""

    def __init__(self, config: BedrockSafetyConfig) -> None:
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        try:
            self.bedrock_runtime_client = create_bedrock_client(self.config)
            self.bedrock_client = create_bedrock_client(self.config, "bedrock")
        except Exception as e:
            raise RuntimeError("Error initializing BedrockSafetyAdapter") from e

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        response = self.bedrock_client.list_guardrails(
            guardrailIdentifier=shield.provider_resource_id,
        )
        if (
            not response["guardrails"]
            or len(response["guardrails"]) == 0
            or response["guardrails"][0]["version"] != shield.params["guardrailVersion"]
        ):
            raise ValueError(
                f"Shield {shield.provider_resource_id} with version {shield.params['guardrailVersion']} not found in Bedrock"
            )

    async def unregister_shield(self, identifier: str) -> None:
        pass

    async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(GetShieldRequest(identifier=request.shield_id))
        if not shield:
            raise ValueError(f"Shield {request.shield_id} not found")

        shield_params = shield.params
        logger.debug("run_shield", shield_params=shield_params, messages=request.messages)

        content_messages = []
        for message in request.messages:
            content_messages.append({"text": {"text": message.content}})
        logger.debug("run_shield final messages", content_messages=json.dumps(content_messages, indent=2))

        response = self.bedrock_runtime_client.apply_guardrail(
            guardrailIdentifier=shield.provider_resource_id,
            guardrailVersion=shield_params["guardrailVersion"],
            source="OUTPUT",
            content=content_messages,
        )
        if response["action"] == "GUARDRAIL_INTERVENED":
            user_message = ""
            metadata = {}
            for output in response["outputs"]:
                user_message = output["text"]
            for assessment in response["assessments"]:
                metadata = dict(assessment)

            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=metadata,
                )
            )

        return RunShieldResponse()
