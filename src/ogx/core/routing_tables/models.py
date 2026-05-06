# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any

from ogx.core.access_control.access_control import is_action_allowed
from ogx.core.datatypes import (
    ModelWithOwner,
    RegistryEntrySource,
)
from ogx.core.request_headers import PROVIDER_DATA_VAR, NeedsRequestProviderData, get_authenticated_user
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import (
    GetModelRequest,
    ListModelsResponse,
    Model,
    ModelNotFoundError,
    Models,
    ModelType,
    OpenAIListModelsResponse,
    OpenAIModel,
    RegisterModelRequest,
    UnregisterModelRequest,
)

from .common import CommonRoutingTableImpl, lookup_model

logger = get_logger(name=__name__, category="core::routing_tables")


class ModelsRoutingTable(CommonRoutingTableImpl, Models):
    """Routing table for managing model registrations, provider lookups, and dynamic model discovery."""

    listed_providers: set[str] = set()

    async def _resolve_auto_model(self, provider_id: str, model_type: ModelType) -> str:
        """Resolve provider_model_id="auto" to an actual model from the provider.

        Queries the provider's list_models() to get truly available models,
        filters by model_type, and returns the first matching model.

        Args:
            provider_id: The provider to query for models
            model_type: The type of model to filter for (llm, embedding, rerank)

        Returns:
            The provider_model_id of a model that matches the criteria

        Raises:
            ValueError: If no suitable model is found for the provider
        """
        if provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider '{provider_id}' not found in routing table")

        provider = self.impls_by_provider_id[provider_id]

        try:
            models = await provider.list_models()
        except Exception as e:
            raise ValueError(
                f"Failed to list models from provider '{provider_id}' for auto model resolution: {e}"
            ) from e

        if not models:
            raise ValueError(f"Provider '{provider_id}' returned no models for auto resolution")

        # Filter by model_type
        matching_models = [m for m in models if m.model_type == model_type]

        if not matching_models:
            raise ValueError(f"No {model_type} models found in provider '{provider_id}' for auto model resolution")

        # Use the first matching model
        # In the future, we could enhance this to:
        # - Prefer recently-used models (likely cached/warm)
        # - Validate availability with a lightweight health check
        selected_model = matching_models[0]
        return selected_model.provider_resource_id

    async def refresh(self) -> None:
        for provider_id, provider in self.impls_by_provider_id.items():
            refresh = await provider.should_refresh_models()
            refresh = refresh or provider_id not in self.listed_providers
            if not refresh:
                continue

            try:
                models = await provider.list_models()
            except Exception as e:
                if provider_id not in self.listed_providers:
                    self.listed_providers.add(provider_id)
                    logger.warning("Model refresh skipped", provider_id=provider_id)
                else:
                    logger.warning("Model refresh failed", provider_id=provider_id, error=str(e))
                continue

            self.listed_providers.add(provider_id)
            if models is None:
                continue

            await self.update_registered_models(provider_id, models)

    async def _get_dynamic_models_from_provider_data(self) -> list[Model]:
        """
        Fetch models from providers that have credentials in the current request's provider_data.

        This allows users to see models available to them from providers that require
        per-request API keys (via X-OGX-Provider-Data header).

        Returns models with fully qualified identifiers (provider_id/model_id) but does NOT
        cache them in the registry since they are user-specific.
        """
        provider_data = PROVIDER_DATA_VAR.get()
        if not provider_data:
            return []

        dynamic_models = []
        user = get_authenticated_user()

        for provider_id, provider in self.impls_by_provider_id.items():
            # Check if this provider supports provider_data
            if not isinstance(provider, NeedsRequestProviderData):
                continue

            # Check if provider has a validator (some providers like ollama don't need per-request credentials)
            spec = getattr(provider, "__provider_spec__", None)
            if not spec or not getattr(spec, "provider_data_validator", None):
                continue

            # Validate provider_data silently - we're speculatively checking all providers
            # so validation failures are expected when user didn't provide keys for this provider
            try:
                validator = instantiate_class_type(spec.provider_data_validator)
                validator(**provider_data)
            except Exception:
                # User didn't provide credentials for this provider - skip silently
                continue

            # Validation succeeded! User has credentials for this provider
            # Now try to list models
            try:
                models = await provider.list_models()
                if not models:
                    continue

                # Ensure models have fully qualified identifiers and apply RBAC filtering
                for model in models:
                    # Only add prefix if model identifier doesn't already have it
                    if not model.identifier.startswith(f"{provider_id}/"):
                        model.identifier = f"{provider_id}/{model.provider_resource_id}"

                    # Convert to ModelWithOwner for RBAC check
                    temp_model = ModelWithOwner(
                        identifier=model.identifier,
                        provider_id=provider_id,
                        provider_resource_id=model.provider_resource_id,
                        model_type=model.model_type,
                        metadata=model.metadata,
                    )

                    # Apply RBAC check - only include models user has read permission for
                    if is_action_allowed(self.policy, "read", temp_model, user):
                        dynamic_models.append(model)
                    else:
                        logger.debug(
                            "Access denied to dynamic model",
                            model=model.identifier,
                            user=user.principal if user else "anonymous",
                        )

                logger.debug(
                    "Fetched accessible models from provider using provider_data",
                    count=len(dynamic_models),
                    provider_id=provider_id,
                )

            except Exception as e:
                logger.debug(
                    "Failed to list models from provider with provider_data", provider_id=provider_id, error=str(e)
                )
                continue

        return dynamic_models

    async def list_models(self) -> ListModelsResponse:
        # Get models from registry
        registry_models = await self.get_all_with_type("model")

        # Get additional models available via provider_data (user-specific, not cached)
        dynamic_models = await self._get_dynamic_models_from_provider_data()

        # Combine, avoiding duplicates (registry takes precedence)
        registry_identifiers = {m.identifier for m in registry_models}
        unique_dynamic_models = [m for m in dynamic_models if m.identifier not in registry_identifiers]

        return ListModelsResponse(data=registry_models + unique_dynamic_models)

    async def openai_list_models(self) -> OpenAIListModelsResponse:
        # Get models from registry
        registry_models = await self.get_all_with_type("model")

        # Get additional models available via provider_data (user-specific, not cached)
        dynamic_models = await self._get_dynamic_models_from_provider_data()

        # Combine, avoiding duplicates (registry takes precedence)
        registry_identifiers = {m.identifier for m in registry_models}
        unique_dynamic_models = [m for m in dynamic_models if m.identifier not in registry_identifiers]

        all_models = registry_models + unique_dynamic_models

        openai_models = [
            OpenAIModel(
                id=model.identifier,
                object="model",
                created=int(time.time()),
                owned_by="ogx",
                custom_metadata={
                    "model_type": model.model_type,
                    "provider_id": model.provider_id,
                    "provider_resource_id": model.provider_resource_id,
                    **model.metadata,
                },
            )
            for model in all_models
        ]
        return OpenAIListModelsResponse(data=openai_models)

    async def get_model(self, request_or_model_id: GetModelRequest | str) -> Model:
        # Support both the public Models API (GetModelRequest) and internal ModelStore interface (string)
        if isinstance(request_or_model_id, GetModelRequest):
            model_id = request_or_model_id.model_id
        else:
            model_id = request_or_model_id
        return await lookup_model(self, model_id)

    async def get_provider_impl(self, model_id: str) -> Any:
        model = await lookup_model(self, model_id)
        if model.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider {model.provider_id} not found in the routing table")
        return self.impls_by_provider_id[model.provider_id]

    async def has_model(self, model_id: str) -> bool:
        """
        Check if a model exists in the routing table.

        :param model_id: The model identifier to check
        :return: True if the model exists, False otherwise
        """
        try:
            await lookup_model(self, model_id)
            return True
        except ModelNotFoundError:
            return False

    async def register_model(
        self,
        request: RegisterModelRequest | str | None = None,
        *,
        model_id: str | None = None,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
        model_validation: bool | None = None,
    ) -> Model:
        # Support both the public Models API (RegisterModelRequest) and legacy parameter-based interface
        if isinstance(request, RegisterModelRequest):
            model_id = request.model_id
            provider_model_id = request.provider_model_id
            provider_id = request.provider_id
            metadata = request.metadata or {}
            model_type = request.model_type
            model_validation = request.model_validation
        elif isinstance(request, str):
            # Legacy positional argument: register_model("model-id", ...)
            model_id = request

        if model_id is None:
            raise ValueError("Either request or model_id must be provided")

        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this model
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    f"Please specify a provider_id for model {model_id} since multiple providers are available: {self.impls_by_provider_id.keys()}.\n\n"
                    "Use the provider_id as a prefix to disambiguate, e.g. 'provider_id/model_id'."
                )

        provider_model_id = provider_model_id or model_id
        metadata = metadata or {}
        model_type = model_type or ModelType.llm
        if "embedding_dimension" not in metadata and model_type == ModelType.embedding:
            raise ValueError("Embedding model must have an embedding dimension in its metadata")

        # Resolve provider_model_id="auto" to an actual model from the provider
        if provider_model_id == "auto":
            provider_model_id = await self._resolve_auto_model(provider_id, model_type)
            logger.info(
                "Resolved auto model alias",
                model_id=model_id,
                provider_id=provider_id,
                resolved_provider_model_id=provider_model_id,
            )

        # Check if this is an unprefixed alias (from provider_id="all" expansion)
        # If so, use model_id directly as identifier without provider prefix
        is_unprefixed = metadata and metadata.get("_unprefixed_alias", False)
        if is_unprefixed:
            identifier = model_id
            # Remove the internal marker from metadata before storing
            metadata = {k: v for k, v in metadata.items() if k != "_unprefixed_alias"}
        else:
            # Avoid double-prefixing if model_id already contains the provider prefix
            if model_id.startswith(f"{provider_id}/"):
                identifier = model_id
            else:
                identifier = f"{provider_id}/{model_id}"

        model = ModelWithOwner(
            identifier=identifier,
            provider_resource_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
            model_validation=model_validation,
            source=RegistryEntrySource.via_register_api,
        )
        registered_model = await self.register_object(model)
        return registered_model

    async def unregister_model(
        self,
        request: UnregisterModelRequest | str | None = None,
        *,
        model_id: str | None = None,
    ) -> None:
        # Support both the public Models API (UnregisterModelRequest) and legacy parameter-based interface
        if isinstance(request, UnregisterModelRequest):
            model_id = request.model_id
        elif isinstance(request, str):
            # Legacy positional argument: unregister_model("model-id")
            model_id = request

        if model_id is None:
            raise ValueError("Either request or model_id must be provided")

        existing_model = await self.get_model(model_id)
        if existing_model is None:
            raise ModelNotFoundError(model_id)
        await self.unregister_object(existing_model)

    async def update_registered_models(
        self,
        provider_id: str,
        models: list[Model],
    ) -> None:
        existing_models = await self.get_all_with_type("model")

        # we may have an alias for the model registered by the user (or during initialization
        # from config.yaml) that we need to keep track of
        model_ids = {}
        for model in existing_models:
            if model.provider_id != provider_id:
                continue
            if model.source == RegistryEntrySource.via_register_api:
                model_ids[model.provider_resource_id] = model.identifier
                continue

            logger.debug("Unregistering model", model=model.identifier)
            await self.unregister_object(model)

        for model in models:
            # Determine what the identifier will be for this provider-listed model
            if model.identifier == model.provider_resource_id:
                model.identifier = f"{provider_id}/{model.provider_resource_id}"

            # Only skip if a user-registered model already has this exact identifier
            # (Different identifiers with the same provider_resource_id can coexist,
            # e.g., "claude-haiku-..." and "vllm/Qwen3-0.6B" both pointing to the same underlying model)
            if model.provider_resource_id in model_ids and model.identifier == model_ids[model.provider_resource_id]:
                logger.debug(
                    "Skipping provider-listed model (user-registered alias exists)",
                    model=model.identifier,
                    provider_resource_id=model.provider_resource_id,
                )
                continue

            logger.debug("Registering model", model=model.identifier, provider_resource_id=model.provider_resource_id)
            await self.register_object(
                ModelWithOwner(
                    identifier=model.identifier,
                    provider_resource_id=model.provider_resource_id,
                    provider_id=provider_id,
                    metadata=model.metadata,
                    model_type=model.model_type,
                    source=RegistryEntrySource.listed_from_provider,
                )
            )
