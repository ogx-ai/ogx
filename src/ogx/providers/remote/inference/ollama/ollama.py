# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import AsyncIterator

import httpx2

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
from ogx.providers.remote.inference.ollama.config import OllamaImplConfig
from ogx.providers.utils.inference.anthropic_mixin import AnthropicMixin
from ogx.providers.utils.inference.models_dev_registry import classify_model
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx.providers.utils.inference.stream_utils import wrap_reasoning_chunks
from ogx.providers.utils.inference.systemone import LOCAL_ENGINE_API_KEY, systemone_judge
from ogx_api import (
    HealthResponse,
    HealthStatus,
    Model,
    ModelType,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionWithReasoning,
    OpenAIMessageParam,
    SystemOneRequest,
    SystemOneResponse,
    UnsupportedModelError,
)

logger = get_logger(name=__name__, category="inference::ollama")

# Library names of Ollama models that serve the SystemOne (decision) protocol natively.
# The OpenAI-compatible /v1/models listing does not expose per-model capabilities, so
# decision models are recognized by their tag-stripped name.
_SYSTEMONE_MODEL_NAMES = frozenset({"clef", "clef-flash", "laya", "nimble", "tev1"})


def _is_systemone_model(identifier: str) -> bool:
    # The engine reports canonical names with a tag (e.g. "laya:latest").
    return identifier.split(":", 1)[0] in _SYSTEMONE_MODEL_NAMES


class OllamaInferenceAdapter(AnthropicMixin, OpenAIMixin):
    """Inference adapter for the Ollama local model runtime."""

    config: OllamaImplConfig

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

    def get_api_key(self):
        return "NO KEY REQUIRED"

    def get_base_url(self):
        return str(self.config.base_url)

    def _anthropic_count_tokens_url(self) -> str | None:
        """Ollama's /v1/messages has no count_tokens endpoint; counting falls through to
        OpenAIMixin's default, which counts by calling anthropic_messages with max_tokens=1."""
        return None

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

        # Ollama expects `reasoning` on assistant messages. Keep the message list typed
        # by rehydrating the remapped assistant message back into the OpenAI assistant model.
        mapped_messages: list[OpenAIMessageParam] = []
        for msg in params.messages:
            if isinstance(msg, AssistantMessageWithReasoning) and msg.reasoning_content:
                msg_dict = msg.model_dump(exclude_none=True)
                msg_dict["reasoning"] = msg_dict.pop("reasoning_content")
                mapped_messages.append(OpenAIAssistantMessageParam.model_validate(msg_dict))
            else:
                mapped_messages.append(msg)
        params.messages = mapped_messages

        result = await self.openai_chat_completion(params)
        # narrow the result type to AsyncIterator for the reasoning wrapper below
        if not isinstance(result, AsyncIterator):
            raise RuntimeError("Expected streaming response for reasoning, but got non-streaming result")

        return wrap_reasoning_chunks(result)

    def _get_ollama_base_url(self) -> str:
        """Get the Ollama base URL without trailing /v1 suffix."""
        base_url_str = str(self.config.base_url)
        if base_url_str.endswith("/v1"):
            return base_url_str[:-3]
        return base_url_str

    def construct_model_from_identifier(self, identifier: str) -> Model:
        model = super().construct_model_from_identifier(identifier)
        if model.model_type == ModelType.llm:
            # Ollama `name:tag` is looked up as `name-tag` because some models.dev providers
            # (e.g. digitalocean, scaleway) list dash-form ids that coincide with Ollama tags, so
            # each size tag keeps its own entry and its own embedding dimension.
            classified = classify_model(identifier.replace(":", "-"), self.__provider_id__)
            # Ollama has no rerank API, so a rerank-named model stays an llm.
            if classified is not None and classified.model_type == ModelType.embedding:
                model.model_type = ModelType.embedding
                model.metadata = classified.metadata
        if _is_systemone_model(identifier):
            # Decision models answer SystemOne judgments, not chat completions; register them
            # under ModelType.systemone so the /v1/systemone router finds them.
            model.model_type = ModelType.systemone
        return model

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        """Answer a SystemOne judgment request by forwarding it to Ollama's /v1/systemone.

        Ollama serves the TypeSafe-compatible SystemOne protocol natively for local decision
        models, so no translation is needed; the shared helper validates the raw response
        body into the OGX wire model. The SDK requires an API key, but Ollama ignores it, so a
        fixed placeholder is passed.
        """
        return await systemone_judge(
            self._get_ollama_base_url(),
            LOCAL_ENGINE_API_KEY,
            request,
            network=self.config.network,
        )

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
        Uses the unauthenticated GET /api/version endpoint.
        Returns:
            HealthResponse: A dictionary containing the health status.
        """
        try:
            url = f"{self._get_ollama_base_url()}/api/version"
            async with httpx2.AsyncClient(**self._build_httpx_client_kwargs(default_timeout=30.0)) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

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
