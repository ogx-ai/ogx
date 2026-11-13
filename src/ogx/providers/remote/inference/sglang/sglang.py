# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import AsyncIterator
from typing import ClassVar

import httpx2
from pydantic import ConfigDict

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
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
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionWithReasoning,
    RerankData,
    RerankResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from ogx_api.inference import RerankRequest

from .config import SGLangInferenceAdapterConfig

log = get_logger(name=__name__, category="inference::sglang")

# Clef and PPLX-Decider use their own decision readout; Tev1 uses SGLang's normal
# next-token scoring path on /v1/systemone. SGLang's /v1/models
# exposes no capabilities and the served id is normally the HuggingFace path, so the last
# path segment is matched by name. SGLang also scores the answer labels of any chat model
# on /v1/systemone; such a model can be registered under ModelType.systemone explicitly.
_SYSTEMONE_MODEL_NAMES = frozenset({"clef", "clef-flash", "clef-omni", "tev1"})
_SYSTEMONE_MODEL_PREFIXES = ("pplx-decider", "tev1-")


def _is_systemone_model(identifier: str) -> bool:
    name = identifier.rsplit("/", 1)[-1].casefold()
    return name in _SYSTEMONE_MODEL_NAMES or name.startswith(_SYSTEMONE_MODEL_PREFIXES)


class SGLangInferenceAdapter(AnthropicMixin, OpenAIMixin):
    """Inference adapter for remote SGLang servers.

    SGLang serves the OpenAI-compatible surface under ``/v1`` (chat completions,
    completions, embeddings, ``/v1/models``, ``/v1/rerank``), the Anthropic
    Messages API under ``/v1/messages`` and an unauthenticated ``/health`` probe.
    """

    config: SGLangInferenceAdapterConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_data_api_key_field: str = "sglang_api_token"
    anthropic_auth_style: ClassVar[str] = "bearer"
    anthropic_no_key_placeholder: ClassVar[str | None] = None

    def get_api_key(self) -> str | None:
        if self.config.auth_credential:
            return self.config.auth_credential.get_secret_value()
        return "NO KEY REQUIRED"

    def get_base_url(self) -> str:
        if not self.config.base_url:
            raise ValueError("No base URL configured")
        return str(self.config.base_url)

    def _get_base_url_without_version(self) -> str:
        """The server root: the configured URL with any trailing ``/v1`` removed."""
        base_url = self.get_base_url().rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url

    async def initialize(self) -> None:
        if not self.config.base_url:
            raise ValueError(
                "You must provide a URL in config.yaml (or via the SGLANG_URL environment variable) to use SGLang."
            )

    async def health(self) -> HealthResponse:
        """Probe SGLang's unauthenticated ``/health`` endpoint."""
        try:
            health_url = self._get_base_url_without_version() + "/health"
            async with httpx2.AsyncClient(**self._build_httpx_client_kwargs()) as client:
                response = await client.get(health_url)
                response.raise_for_status()
                return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        params = params.model_copy()
        if params.max_tokens is None and self.config.max_tokens:
            params.max_tokens = self.config.max_tokens
        return await super().openai_chat_completion(params)

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Chat completion with reasoning support for SGLang.

        SGLang streams the model's thinking as ``delta.reasoning_content`` and
        reads prior-turn thinking back from ``reasoning_content`` on assistant
        messages, so the history is passed through unchanged.
        """
        if not params.stream:
            raise NotImplementedError("Non-streaming reasoning is not yet supported for SGLang")

        params = params.model_copy()
        # Dump the assistant messages that carry reasoning to dicts so the
        # `reasoning_content` key reaches SGLang as-is (the typed message drops it).
        mapped_messages: list = [
            msg.model_dump(exclude_none=True) if isinstance(msg, AssistantMessageWithReasoning) else msg
            for msg in params.messages
        ]
        params.messages = mapped_messages

        result = await self.openai_chat_completion(params)
        if not isinstance(result, AsyncIterator):
            raise RuntimeError("Expected streaming response for reasoning, but got non-streaming result")
        return wrap_reasoning_chunks(result)

    def construct_model_from_identifier(self, identifier: str) -> Model:
        # SGLang's /v1/models response does not expose a model task/type field,
        # so we classify with models.dev with a name fallback.
        model = classify_model(identifier, self.__provider_id__)
        if model is None:
            model = super().construct_model_from_identifier(identifier)
        if _is_systemone_model(identifier):
            # Decision models answer SystemOne judgments, not chat completions; register them
            # under ModelType.systemone so the /v1/systemone router finds them.
            model.model_type = ModelType.systemone
        return model

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        """Answer a SystemOne judgment request by forwarding it to SGLang's /v1/systemone.

        SGLang serves the TypeSafe-compatible SystemOne protocol natively, through the
        decision readout of a decision checkpoint or by scoring the answer labels of a chat
        model, so no translation is needed; the shared helper validates the raw response
        body into the OGX wire model.
        """
        # get_api_key() returns a display placeholder ("NO KEY REQUIRED") when no
        # credential is configured, which the typesafe SDK rejects; pass the
        # configured credential or the fixed local placeholder instead.
        api_key = (
            self.config.auth_credential.get_secret_value()
            if self.config.auth_credential is not None
            else LOCAL_ENGINE_API_KEY
        )
        return await systemone_judge(
            self._get_base_url_without_version(),
            api_key,
            request,
            network=self.config.network,
            # SGLang adds the full-vocabulary probability of the answer labels to every
            # answer as `x_label_mass`, an SGLang extension outside the SystemOne schema.
            extension_fields=frozenset({"x_label_mass"}),
        )

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        def format_item(
            item: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam,
        ) -> str:
            if isinstance(item, str):
                return item
            elif isinstance(item, OpenAIChatCompletionContentPartTextParam):
                return item.text
            elif isinstance(item, OpenAIChatCompletionContentPartImageParam):
                raise ValueError("SGLang rerank API does not support images")
            else:
                raise ValueError("Unsupported item type for reranking")

        # SGLang's rerank request has no model field: the server reranks with the
        # model it was launched with, so the requested model is not sent.
        payload: dict[str, str | int | float | list[str]] = {
            "query": format_item(request.query),
            "documents": [format_item(item) for item in request.items],
        }
        if request.max_num_results is not None:
            payload["top_n"] = request.max_num_results

        # SGLang serves reranking at /v1/rerank and answers with a list of
        # {"score": float, "index": int, ...} entries, one per document.
        endpoint = self._get_base_url_without_version() + "/v1/rerank"

        headers: dict[str, str] = {}
        api_key = self._get_api_key_from_config_or_provider_data()
        if api_key and api_key != "NO KEY REQUIRED":
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx2.AsyncClient(**self._build_httpx_client_kwargs()) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"SGLang rerank API request failed with status {response.status_code}: {response.text}"
                    )

                def convert_result_item(item: dict) -> RerankData:
                    if "index" not in item or "score" not in item:
                        raise RuntimeError("SGLang rerank API response missing required fields 'index' or 'score'")
                    try:
                        return RerankData(index=int(item["index"]), relevance_score=float(item["score"]))
                    except (TypeError, ValueError) as e:
                        raise RuntimeError(f"Invalid data types in SGLang rerank API response: {e}") from e

                result = response.json()
                if not isinstance(result, list):
                    raise RuntimeError("SGLang rerank API response is not a list of results")

                rerank_data = [convert_result_item(item) for item in result]
                rerank_data.sort(key=lambda entry: entry.relevance_score, reverse=True)
                return RerankResponse(data=rerank_data)

        except httpx2.HTTPError as e:
            raise ConnectionError(f"Failed to connect to SGLang rerank API at {endpoint}: {e}") from e
