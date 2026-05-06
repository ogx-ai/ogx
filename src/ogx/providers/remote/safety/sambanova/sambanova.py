# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
import litellm

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
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

from .config import SambaNovaSafetyConfig

logger = get_logger(name=__name__, category="safety::sambanova")

CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"


class SambaNovaSafetyAdapter(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate, NeedsRequestProviderData):
    """Safety adapter for content moderation using SambaNova AI services."""

    def __init__(self, config: SambaNovaSafetyConfig) -> None:
        self.config = config
        self.environment_available_models = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_api_key(self) -> str:
        config_api_key = self.config.api_key if self.config.api_key else None
        if config_api_key:
            return config_api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.sambanova_api_key:
                raise ValueError(
                    'Pass Sambanova API Key in the header X-OGX-Provider-Data as { "sambanova_api_key": <your api key> }'
                )
            return provider_data.sambanova_api_key

    async def register_shield(self, shield: Shield) -> None:
        list_models_url = self.config.url + "/models"
        if len(self.environment_available_models) == 0:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                    response = await client.get(list_models_url)
                    response.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"Request to {list_models_url} failed") from e
            self.environment_available_models = [model.get("id") for model in response.json().get("data", {})]
        if (
            "guard" not in shield.provider_resource_id.lower()
            or shield.provider_resource_id.split("sambanova/")[-1] not in self.environment_available_models
        ):
            logger.warning(
                "Shield not available in",
                provider_resource_id=shield.provider_resource_id,
                list_models_url=list_models_url,
            )

    async def unregister_shield(self, identifier: str) -> None:
        pass

    async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(GetShieldRequest(identifier=request.shield_id))
        if not shield:
            raise ValueError(f"Shield {request.shield_id} not found")

        shield_params = shield.params
        logger.debug("run_shield", shield_params=shield_params, messages=request.messages)

        response = await litellm.acompletion(
            model=shield.provider_resource_id,
            messages=request.messages,
            api_key=self._get_api_key(),
        )
        shield_message = response.choices[0].message.content

        if "unsafe" in shield_message.lower():
            user_message = CANNED_RESPONSE_TEXT
            violation_type = shield_message.split("\n")[-1]
            metadata = {"violation_type": violation_type}

            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=metadata,
                )
            )

        return RunShieldResponse()
