# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx.providers.utils.forward_headers import build_forwarded_headers
from ogx.providers.utils.inference.stream_utils import wrap_async_stream
from ogx_api import (
    Inference,
    Model,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

from .config import PassthroughImplConfig

logger = get_logger(__name__, category="inference")


class PassthroughInferenceAdapter(NeedsRequestProviderData, Inference):
    """Inference adapter that forwards requests to any OpenAI-compatible endpoint."""

    def __init__(self, config: PassthroughImplConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        return model

    async def list_models(self) -> list[Model]:
        """List models by calling the downstream /v1/models endpoint."""
        client = self._get_openai_client()

        response = await client.models.list()

        # Convert from OpenAI format to OGX Model format
        models = []
        for model_data in response.data:
            downstream_model_id = model_data.id
            custom_metadata = getattr(model_data, "custom_metadata", {}) or {}

            # Prefix identifier with provider ID for local registry
            local_identifier = f"{self.__provider_id__}/{downstream_model_id}"

            model = Model(
                identifier=local_identifier,
                provider_id=self.__provider_id__,
                provider_resource_id=downstream_model_id,
                model_type=custom_metadata.get("model_type", "llm"),
                metadata=custom_metadata,
            )
            models.append(model)

        return models

    async def should_refresh_models(self) -> bool:
        """Passthrough should refresh models since they come from downstream dynamically."""
        return self.config.refresh_models

    def _get_openai_client(self) -> AsyncOpenAI:
        """Get an AsyncOpenAI client configured for the downstream server."""
        base_url = self._get_passthrough_url()
        request_headers = self._build_request_headers()

        # api_key="" means the SDK adds no Authorization header of its own;
        # auth comes entirely from request_headers (forwarded or static api_key).
        # This avoids the "passthrough" sentinel that would send a spurious
        # Authorization: Bearer passthrough to every downstream, even when
        # forward_headers only targets non-auth headers like X-Tenant-ID.
        return AsyncOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key="",
            default_headers=request_headers or None,
        )

    def _build_request_headers(self) -> dict[str, str]:
        """Build outbound headers: forwarded provider-data keys first, then static api_key.

        Static api_key always wins over a forwarded Authorization, regardless of casing.
        """
        provider_data = self.get_request_provider_data()
        headers = build_forwarded_headers(provider_data, self.config.forward_headers)
        if self.config.forward_headers and not headers:
            logger.warning(
                "forward_headers is configured but no matching keys found in provider data — "
                "outbound request may be unauthenticated"
            )
        api_key = self._get_passthrough_api_key_or_none(provider_data)
        if api_key:
            # remove any forwarded authorization variant (case-insensitive) so static key wins
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get_passthrough_api_key_or_none(self, provider_data: object | None = None) -> str | None:
        """Return the static or per-request API key, or None if not configured."""
        if self.config.auth_credential is not None:
            configured_api_key = self.config.auth_credential.get_secret_value()
            if configured_api_key:
                return configured_api_key

        if provider_data is None:
            provider_data = self.get_request_provider_data()
        passthrough_api_key = getattr(provider_data, "passthrough_api_key", None)
        if passthrough_api_key is not None:
            if hasattr(passthrough_api_key, "get_secret_value"):
                provider_data_api_key = passthrough_api_key.get_secret_value()
            else:
                provider_data_api_key = str(passthrough_api_key)
            if provider_data_api_key:
                return provider_data_api_key

        return None

    def _get_passthrough_url(self) -> str:
        """Get the passthrough URL from config or provider data."""
        if self.config.base_url is not None:
            return str(self.config.base_url)

        provider_data = self.get_request_provider_data()
        if provider_data is None or provider_data.passthrough_url is None:
            raise ValueError(
                'Pass url of the passthrough endpoint in the header X-OGX-Provider-Data as { "passthrough_url": <your passthrough url>}'
            )
        return str(provider_data.passthrough_url)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Forward completion request to downstream using OpenAI client."""
        client = self._get_openai_client()
        request_params = params.model_dump(exclude_none=True)
        response = await client.completions.create(**request_params)

        if params.stream:
            return wrap_async_stream(response)

        return response  # type: ignore[return-value]

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Forward chat completion request to downstream using OpenAI client."""
        client = self._get_openai_client()
        request_params = params.model_dump(exclude_none=True)
        response = await client.chat.completions.create(**request_params)

        if params.stream:
            return wrap_async_stream(response)

        return response  # type: ignore[return-value]

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Forward embeddings request to downstream using OpenAI client."""
        client = self._get_openai_client()
        request_params = params.model_dump(exclude_none=True)
        response = await client.embeddings.create(**request_params)
        return response  # type: ignore
