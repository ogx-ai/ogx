# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import asyncio
from collections.abc import AsyncIterator

from ollama import AsyncClient as AsyncOllamaClient

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
from ogx.providers.remote.inference.ollama.config import OllamaImplConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    HealthResponse,
    HealthStatus,
    Model,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionWithReasoning,
    UnsupportedModelError,
)

logger = get_logger(name=__name__, category="inference::ollama")


class OllamaInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Ollama local model runtime."""

    config: OllamaImplConfig

    # automatically set by the resolver when instantiating the provider
    __provider_id__: str

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "all-minilm:l6-v2": {
            "embedding_dimension": 384,
            "context_length": 512,
        },
        "nomic-embed-text:latest": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
        "nomic-embed-text:v1.5": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
        "nomic-embed-text:137m-v1.5-fp16": {
            "embedding_dimension": 768,
            "context_length": 8192,
        },
    }

    download_images: bool = True
    _clients: dict[asyncio.AbstractEventLoop, AsyncOllamaClient] = {}

    @property
    def ollama_client(self) -> AsyncOllamaClient:
        # ollama client attaches itself to the current event loop (sadly?)
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            # Ollama client expects base URL without /v1 suffix
            base_url_str = str(self.config.base_url)
            if base_url_str.endswith("/v1"):
                host = base_url_str[:-3]
            else:
                host = base_url_str
            self._clients[loop] = AsyncOllamaClient(host=host)
        return self._clients[loop]

    def get_api_key(self):
        return "NO KEY REQUIRED"

    def get_base_url(self):
        return str(self.config.base_url)

    def _prepare_reasoning_params(self, params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        """Adapt CC request params to match what Ollama expects for reasoning.

        Each provider may need different param adjustments. For Ollama:
        - If reasoning_effort is not set, default to "none" so Ollama
          doesn't apply its own default (medium).

        Override this in other providers if they need different mapping,
        e.g. converting effort levels to boolean flags.
        """
        if params.reasoning_effort is None:
            params.reasoning_effort = "none"

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Chat completion with reasoning support for Ollama.

        Extracts reasoning from Ollama's response and wraps it in internal
        types so the Responses layer can read reasoning as a typed field.
        """
        if not params.stream:
            raise NotImplementedError("Non-streaming reasoning is not yet supported for Ollama")

        params = params.model_copy()
        self._prepare_reasoning_params(params)

        # Ollama's CC endpoint expects 'reasoning' on assistant messages, but
        # that field isn't part of the official CC spec. Convert to dicts so we
        # can rename reasoning_content → reasoning.
        mapped_messages: list = []
        for msg in params.messages:
            if isinstance(msg, AssistantMessageWithReasoning) and msg.reasoning_content:
                msg_dict = msg.model_dump(exclude_none=True)
                msg_dict["reasoning"] = msg_dict.pop("reasoning_content")
                mapped_messages.append(msg_dict)
            else:
                mapped_messages.append(msg)
        params.messages = mapped_messages

        result = await self.openai_chat_completion(params)

        async def _wrap_chunks() -> AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
            async for chunk in result:
                reasoning = None
                for choice in chunk.choices or []:
                    reasoning = getattr(choice.delta, "reasoning", None) or getattr(
                        choice.delta, "reasoning_content", None
                    )
                yield OpenAIChatCompletionChunkWithReasoning(
                    chunk=chunk,
                    reasoning_content=reasoning,
                )

        return _wrap_chunks()

    async def initialize(self) -> None:
        logger.info("checking connectivity to Ollama", base_url=self.config.base_url)
        r = await self.health()
        if r["status"] == HealthStatus.ERROR:
            logger.warning(
                "Ollama Server is not running (message: ). Make sure to start it using `ollama serve` in a separate terminal",
                r_message=r["message"],
            )

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the Ollama server.
        This method is used by initialize() and the Provider API to verify that the service is running
        correctly.
        Returns:
            HealthResponse: A dictionary containing the health status.
        """
        try:
            await self.ollama_client.ps()
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def shutdown(self) -> None:
        self._clients.clear()

    async def register_model(self, model: Model) -> Model:
        if await self.check_model_availability(model.provider_model_id):
            return model
        elif await self.check_model_availability(f"{model.provider_model_id}:latest"):
            model.provider_resource_id = f"{model.provider_model_id}:latest"
            logger.warning(
                "Imprecise provider resource id was used but 'latest' is available in Ollama - using",
                provider_model_id=model.provider_model_id,
            )
            return model

        raise UnsupportedModelError(model.provider_model_id, list(self._model_cache.keys()))
