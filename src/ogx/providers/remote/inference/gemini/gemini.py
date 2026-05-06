# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    validate_embeddings_input_is_text,
)

from .config import GeminiConfig


class GeminiInferenceAdapter(OpenAIMixin):
    """Inference adapter for Google Gemini models."""

    config: GeminiConfig

    provider_data_api_key_field: str = "gemini_api_key"
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "models/text-embedding-004": {"embedding_dimension": 768, "context_length": 2048},
        "models/gemini-embedding-001": {"embedding_dimension": 3072, "context_length": 2048},
    }

    def get_base_url(self):
        return "https://generativelanguage.googleapis.com/v1beta/openai/"

    def get_api_key(self) -> str | None:
        """Return API key or access token for the OpenAI-compatible client.

        The AsyncOpenAI client sends this as ``Authorization: Bearer <value>``,
        which Google's OpenAI-compatible endpoint accepts for both API keys and
        OAuth access tokens.  ``access_token`` takes precedence when set.
        """
        if self.config.access_token:
            return self.config.access_token.get_secret_value()
        return super().get_api_key()

    def get_extra_client_params(self) -> dict[str, Any]:
        """Pass quota-project header when using OAuth/ADC credentials."""
        if self.config.project:
            return {"default_headers": {"x-goog-user-project": self.config.project}}
        return {}

    def get_passthrough_auth_headers(self) -> dict[str, str]:
        """Return auth headers appropriate for native Google API passthrough.

        API keys use ``x-goog-api-key``; OAuth tokens use ``Authorization: Bearer``.
        When a project is configured, includes ``x-goog-user-project`` for quota.
        """
        headers: dict[str, str] = {}
        if self.config.project:
            headers["x-goog-user-project"] = self.config.project
        if self.config.access_token:
            headers["Authorization"] = f"Bearer {self.config.access_token.get_secret_value()}"
        else:
            api_key = self._get_api_key_from_config_or_provider_data()
            if api_key:
                headers["x-goog-api-key"] = api_key
        return headers

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """
        Override embeddings method to handle Gemini's missing usage statistics.
        Gemini's embedding API doesn't return usage information, so we provide default values.
        """
        # Validate that input contains only text, not token arrays
        validate_embeddings_input_is_text(params)

        # Build request params conditionally to avoid NotGiven/Omit type mismatch
        request_params: dict[str, Any] = {
            "model": await self._get_provider_model_id(params.model),
            "input": params.input,
        }
        if params.encoding_format is not None:
            request_params["encoding_format"] = params.encoding_format
        if params.dimensions is not None:
            request_params["dimensions"] = params.dimensions
        if params.user is not None:
            request_params["user"] = params.user
        if params.model_extra:
            request_params["extra_body"] = params.model_extra

        response = await self.client.embeddings.create(**request_params)

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        # Gemini doesn't return usage statistics - use default values
        if hasattr(response, "usage") and response.usage:
            usage = OpenAIEmbeddingUsage(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            usage = OpenAIEmbeddingUsage(
                prompt_tokens=0,
                total_tokens=0,
            )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=params.model,
            usage=usage,
        )
