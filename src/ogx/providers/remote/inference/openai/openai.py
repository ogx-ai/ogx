# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.log import get_logger
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    Model,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
)

from .config import OpenAIConfig

logger = get_logger(name=__name__, category="inference::openai")

# Max output tokens per OpenAI model. OpenAI's /v1/models endpoint does not
# expose this, so we maintain the mapping statically.
_MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "gpt-4.1": 32768,
    "gpt-4.1-mini": 32768,
    "gpt-4.1-nano": 32768,
    "gpt-4o": 16384,
    "gpt-4o-mini": 16384,
    "gpt-4-turbo": 4096,
    "gpt-4": 8192,
    "o1": 100000,
    "o1-mini": 65536,
    "o1-pro": 100000,
    "o3": 100000,
    "o3-mini": 100000,
    "o3-pro": 100000,
    "o4-mini": 100000,
}

_WARNED_MODELS: set[str] = set()


#
# This OpenAI adapter implements Inference methods using OpenAIMixin
#
class OpenAIInferenceAdapter(OpenAIMixin):
    """
    OpenAI Inference Adapter for OGX.
    """

    config: OpenAIConfig

    provider_data_api_key_field: str = "openai_api_key"

    supports_tokenized_embeddings_input: bool = True

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-3-large": {"embedding_dimension": 3072, "context_length": 8192},
    }

    def _get_max_output_tokens(self, model: str) -> int | None:
        if model in _MODEL_MAX_OUTPUT_TOKENS:
            return _MODEL_MAX_OUTPUT_TOKENS[model]

        # Try prefix matching for dated snapshot variants (e.g. gpt-4o-2024-08-06)
        for base_model, limit in sorted(
            _MODEL_MAX_OUTPUT_TOKENS.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if model.startswith(f"{base_model}-"):
                return limit

        if model not in _WARNED_MODELS:
            _WARNED_MODELS.add(model)
            logger.warning(
                "Unknown max_output_tokens for model, requests will not be clamped",
                model=model,
            )
        return None

    def construct_model_from_identifier(self, identifier: str) -> Model:
        if metadata := self.embedding_model_metadata.get(identifier):
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.embedding,
                metadata=metadata,
            )

        metadata = {}
        max_output_tokens = self._get_max_output_tokens(identifier)
        if max_output_tokens is not None:
            metadata["max_output_tokens"] = max_output_tokens

        return Model(
            provider_id=self.__provider_id__,  # type: ignore[attr-defined]
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.llm,
            metadata=metadata,
        )

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        max_output_tokens = self._get_max_output_tokens(params.model)
        if max_output_tokens is not None:
            updated_params = params
            if params.max_tokens is not None and params.max_tokens > max_output_tokens:
                updated_params = updated_params.model_copy()
                updated_params.max_tokens = max_output_tokens
            if params.max_completion_tokens is not None and params.max_completion_tokens > max_output_tokens:
                if updated_params is params:
                    updated_params = updated_params.model_copy()
                updated_params.max_completion_tokens = max_output_tokens
            params = updated_params

        return await super().openai_chat_completion(params)

    async def openai_chat_completions_with_reasoning(self, params) -> None:
        raise ValueError(
            "OpenAI provider does not support reasoning. "
            "Please remove the reasoning parameter from your request or choose another provider."
        )

    def get_base_url(self) -> str:
        """
        Get the OpenAI API base URL.

        Returns the OpenAI API base URL from the configuration.
        """
        return str(self.config.base_url)
