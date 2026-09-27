# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from ogx.log import get_logger
from ogx.providers.utils.inference.network_config import (
    LimitsConfig,
    NetworkConfig,
    ProxyConfig,
    TimeoutConfig,
    TLSConfig,
)
from ogx_api import Model, ModelsProtocolPrivate, ModelType, UnsupportedModelError

logger = get_logger(name=__name__, category="providers::utils")


# Re-export network config classes for backward compatibility.
__all__ = [
    "LimitsConfig",
    "NetworkConfig",
    "ProxyConfig",
    "TLSConfig",
    "TimeoutConfig",
]


class RemoteInferenceProviderConfig(BaseModel):
    """Base configuration for remote inference providers with model filtering and auth settings."""

    allowed_models: list[str] | None = Field(
        default=None,
        description="List of models that should be registered with the model registry. If None, all models are allowed.",
    )
    refresh_models: bool = Field(
        default=False,
        description="Whether to refresh models periodically from the provider",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        description="Authentication credential for the provider",
        alias="api_key",
    )
    network: NetworkConfig | None = Field(
        default=None,
        description="Network configuration including TLS, proxy, and timeout settings.",
    )


# TODO: this class is more confusing than useful right now. We need to make it
# more closer to the Model class.
class ProviderModelEntry(BaseModel):
    """Describes a model available from a provider with its aliases and metadata."""

    provider_model_id: str
    aliases: list[str] = Field(default_factory=list)
    model_type: ModelType = ModelType.llm
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRegistryHelper(ModelsProtocolPrivate):
    """Manages model registration, alias resolution, and availability checks for a provider."""

    __provider_id__: str

    def __init__(
        self,
        model_entries: list[ProviderModelEntry] | None = None,
        allowed_models: list[str] | None = None,
    ):
        self.allowed_models = allowed_models if allowed_models else []

        self.alias_to_provider_id_map = {}
        self.model_entries = model_entries or []
        for entry in self.model_entries:
            for alias in entry.aliases:
                self.alias_to_provider_id_map[alias] = entry.provider_model_id

            # also add a mapping from provider model id to itself for easy lookup
            self.alias_to_provider_id_map[entry.provider_model_id] = entry.provider_model_id

    async def list_models(self) -> list[Model] | None:
        models = []
        for entry in self.model_entries:
            ids = [entry.provider_model_id] + entry.aliases
            for id in ids:
                if self.allowed_models and id not in self.allowed_models:
                    continue
                models.append(
                    Model(
                        identifier=id,
                        provider_resource_id=entry.provider_model_id,
                        model_type=entry.model_type,
                        metadata=entry.metadata,
                        provider_id=self.__provider_id__,
                    )
                )
        return models

    async def should_refresh_models(self) -> bool:
        return False

    def get_provider_model_id(self, identifier: str) -> str | None:
        return self.alias_to_provider_id_map.get(identifier, None)

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from the provider (non-static check).

        This is for subclassing purposes, so providers can check if a specific
        model is currently available for use through dynamic means (e.g., API calls).

        This method should NOT check statically configured model entries in
        `self.alias_to_provider_id_map` - that is handled separately in register_model.

        Default implementation returns False (no dynamic models available).

        :param model: The model identifier to check.
        :return: True if the model is available dynamically, False otherwise.
        """
        logger.info(
            "check_model_availability is not implemented for . Returning False by default.",
            __name__=self.__class__.__name__,
        )
        return False

    async def register_model(self, model: Model) -> Model:
        # Check if model is supported in static configuration
        supported_model_id = self.get_provider_model_id(model.provider_resource_id)

        # If not found in static config, check if it's available dynamically from provider
        if not supported_model_id:
            if await self.check_model_availability(model.provider_resource_id):
                supported_model_id = model.provider_resource_id
            else:
                # note: we cannot provide a complete list of supported models without
                #       getting a complete list from the provider, so we return "..."
                all_supported_models = [*self.alias_to_provider_id_map.keys(), "..."]
                raise UnsupportedModelError(model.provider_resource_id, all_supported_models)

        provider_resource_id = self.get_provider_model_id(model.model_id)
        if model.model_type == ModelType.embedding:
            # embedding models are always registered by their provider model id and does not need to be mapped to a llama model
            provider_resource_id = model.provider_resource_id
        if provider_resource_id and provider_resource_id != supported_model_id:
            raise ValueError(
                f"Model id '{model.model_id}' is already registered. Please use a different id or unregister it first."
            )

        # Register the model alias, ensuring it maps to the correct provider model id
        self.alias_to_provider_id_map[model.model_id] = supported_model_id

        return model

    async def unregister_model(self, model_id: str) -> None:
        # model_id is the identifier, not the provider_resource_id
        # unfortunately, this ID can be of the form provider_id/model_id which
        # we never registered. TODO: fix this by significantly rewriting
        # registration and registry helper
        pass
