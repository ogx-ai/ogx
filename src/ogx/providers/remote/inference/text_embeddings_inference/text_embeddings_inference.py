# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable

from ogx.log import get_logger
from ogx.providers.remote.inference.text_embeddings_inference.config import TextEmbeddingsInferenceConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx.providers.utils.inference.server_signature import (
    ServerUnreachableError,
    check_text_embeddings_inference_server,
    get_text_embeddings_inference_model_id,
    verify_text_embeddings_inference_server,
)
from ogx_api import (
    HealthResponse,
    Model,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
)

logger = get_logger(name=__name__, category="inference::text_embeddings_inference")


class TextEmbeddingsInferenceAdapter(OpenAIMixin):
    """Inference adapter for HuggingFace Text-Embeddings-Inference servers.

    TEI servers only serve embedding models, so chat and completion endpoints
    are not supported. The single served model is discovered from the server's
    native /info endpoint (TEI has no /v1/models endpoint) and registered as
    an embedding model.
    """

    config: TextEmbeddingsInferenceConfig

    def get_api_key(self) -> str | None:
        # TEI servers do not authenticate by default; an API key is only needed
        # when the server is fronted by an authenticating proxy.
        if self.config.auth_credential is None:
            return "NO KEY REQUIRED"
        return self.config.auth_credential.get_secret_value()

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    def _signature_api_key(self) -> str | None:
        return self.config.auth_credential.get_secret_value() if self.config.auth_credential else None

    async def initialize(self) -> None:
        # Fail fast at construction time if the server is running but is not a
        # Text-Embeddings-Inference server. A server that is simply not up yet
        # only warns, matching the Ollama adapter's behaviour.
        try:
            await verify_text_embeddings_inference_server(
                self.get_base_url(), api_key=self._signature_api_key(), client_kwargs=self._build_httpx_client_kwargs()
            )
        except ServerUnreachableError as e:
            logger.warning(
                "Text-Embeddings-Inference server is not running; it must be reachable before models can be listed",
                base_url=self.get_base_url(),
                error=str(e),
            )

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying the server identifies as
        Text-Embeddings-Inference via its native GET /info endpoint.
        This method is used by the Provider API to verify that the service is
        running correctly.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        return await check_text_embeddings_inference_server(
            self.get_base_url(), api_key=self._signature_api_key(), client_kwargs=self._build_httpx_client_kwargs()
        )

    def construct_model_from_identifier(self, identifier: str) -> Model:
        # TEI only hosts embedding models, so every discovered model is an embedding model
        return Model(
            provider_id=self.__provider_id__,
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.embedding,
        )

    async def list_provider_model_ids(self) -> Iterable[str]:
        # TEI has no /v1/models endpoint; the single served model is reported
        # by its native GET /info endpoint.
        model_id = await get_text_embeddings_inference_model_id(
            self.get_base_url(), api_key=self._signature_api_key(), client_kwargs=self._build_httpx_client_kwargs()
        )
        return [model_id]

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError("Text-Embeddings-Inference provider does not support chat completions")

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        raise NotImplementedError("Text-Embeddings-Inference provider does not support completions")
