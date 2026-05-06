# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.log import get_logger
from ogx.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from ogx_api import (
    InferenceProvider,
    Model,
    ModelsProtocolPrivate,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
)

from .config import SentenceTransformersInferenceConfig

log = get_logger(name=__name__, category="inference")


class SentenceTransformersInferenceImpl(
    SentenceTransformerEmbeddingMixin,
    InferenceProvider,
    ModelsProtocolPrivate,
):
    """Inference provider for text embeddings using sentence-transformers models."""

    __provider_id__: str

    def __init__(self, config: SentenceTransformersInferenceConfig) -> None:
        self.config = config

    async def openai_chat_completions_with_reasoning(self, params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        raise NotImplementedError("SentenceTransformers provider does not support reasoning in chat completions")

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def should_refresh_models(self) -> bool:
        return False

    async def list_models(self) -> list[Model] | None:
        return [
            Model(
                identifier="nomic-ai/nomic-embed-text-v1.5",
                provider_resource_id="nomic-ai/nomic-embed-text-v1.5",
                provider_id=self.__provider_id__,
                metadata={
                    "embedding_dimension": 768,
                },
                model_type=ModelType.embedding,
            ),
        ]

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion not supported by sentence transformers provider")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError("OpenAI chat completion not supported by sentence transformers provider")
