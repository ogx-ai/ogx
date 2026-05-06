# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import AsyncIterator
from urllib.parse import urljoin

import httpx
from pydantic import ConfigDict

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
from ogx.providers.utils.inference.http_client import (
    build_network_client_kwargs as _build_network_client_kwargs,
)
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
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
)
from ogx_api.inference import RerankRequest

from .config import VLLMInferenceAdapterConfig

log = get_logger(name=__name__, category="inference::vllm")


class VLLMInferenceAdapter(OpenAIMixin):
    """Inference adapter for remote vLLM servers."""

    config: VLLMInferenceAdapterConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_data_api_key_field: str = "vllm_api_token"

    def get_api_key(self) -> str | None:
        if self.config.auth_credential:
            return self.config.auth_credential.get_secret_value()
        return "NO KEY REQUIRED"

    def get_base_url(self) -> str:
        """Get the base URL from config."""
        if not self.config.base_url:
            raise ValueError("No base URL configured")
        return str(self.config.base_url)

    async def initialize(self) -> None:
        if not self.config.base_url:
            raise ValueError(
                "You must provide a URL in config.yaml (or via the VLLM_URL environment variable) to use vLLM."
            )

    def _build_httpx_client_kwargs(self) -> dict:
        """Build httpx.AsyncClient kwargs that honour network/TLS configuration."""
        kwargs = _build_network_client_kwargs(self.config.network)
        if not kwargs:
            kwargs["verify"] = self.shared_ssl_context
        return kwargs

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the remote vLLM server.
        This method is used by the Provider API to verify
        that the service is running correctly.
        Uses the unauthenticated /health endpoint.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            base_url = self.get_base_url()
            health_url = urljoin(base_url, "health")

            async with httpx.AsyncClient(**self._build_httpx_client_kwargs()) as client:
                response = await client.get(health_url)
                response.raise_for_status()
                return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def check_model_availability(self, model: str) -> bool:
        """
        Skip the check when running without authentication.
        """
        if not self.config.auth_credential:
            model_ids = []
            async for m in self.client.models.list():
                if m.id == model:  # Found exact match
                    return True
                model_ids.append(m.id)
            raise ValueError(f"Model '{model}' not found. Available models: {model_ids}")
        log.warning(f"Not checking model availability for {model} as API token may trigger OAuth workflow")
        return True

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        params = params.model_copy()

        # Apply vLLM-specific defaults
        if params.max_tokens is None and self.config.max_tokens:
            params.max_tokens = self.config.max_tokens

        return await super().openai_chat_completion(params)

    def _prepare_reasoning_params(self, params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        """Adapt CC request params to match what vLLM expects for reasoning.

        No-op for now. Override if vLLM needs specific param adjustments,
        e.g. mapping effort levels or moving params to extra_body.
        """
        pass

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Chat completion with reasoning support for vLLM.

        Extracts reasoning from vLLM's response and wraps it in internal
        types (OpenAIChatCompletionChunkWithReasoning / OpenAIChatCompletionWithReasoning)
        so the Responses layer can read reasoning as a typed field.
        """
        if not params.stream:
            raise NotImplementedError("Non-streaming reasoning is not yet supported for vLLM")

        params = params.model_copy()
        self._prepare_reasoning_params(params)

        # vLLM's CC endpoint expects 'reasoning' on assistant messages, but
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
            async for chunk in result:  # type: ignore[union-attr]
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

    def construct_model_from_identifier(self, identifier: str) -> Model:
        # vLLM's /v1/models response does not expose a model task/type field, so classify by name.
        if "embed" in identifier.lower():
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.embedding,
                metadata={},
            )
        if "rerank" in identifier.lower():
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.rerank,
            )
        return super().construct_model_from_identifier(identifier)

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
                raise ValueError("vLLM rerank API does not support images")
            else:
                raise ValueError("Unsupported item type for reranking")

        payload: dict[str, str | int | float | list[str]] = {
            "model": request.model,
            "query": format_item(request.query),
            "documents": [format_item(item) for item in request.items],
        }
        if request.max_num_results is not None:
            payload["top_n"] = request.max_num_results

        # vLLM does not support /v1/rerank ->
        #   "To indicate that the rerank API is not part of the standard OpenAI API,
        #    we have located it at `/rerank`. Please update your client accordingly.
        #    (Note: Conforms to JinaAI rerank API)" - vLLM 0.15.1
        endpoint = self.get_base_url().replace("/v1", "") + "/rerank"  # TODO: find a better solution

        headers: dict[str, str] = {}
        api_key = self._get_api_key_from_config_or_provider_data()
        if api_key and api_key != "NO KEY REQUIRED":
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(**self._build_httpx_client_kwargs()) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"vLLM rerank API request failed with status {response.status_code}: {response.text}"
                    )

                def convert_result_item(item: dict) -> RerankData:
                    if "index" not in item or "relevance_score" not in item:
                        raise RuntimeError(
                            "vLLM rerank API response missing required fields 'index' or 'relevance_score'"
                        )

                    try:
                        return RerankData(index=int(item["index"]), relevance_score=float(item["relevance_score"]))
                    except (TypeError, ValueError) as e:
                        raise RuntimeError(f"Invalid data types in vLLM rerank API response: {e}") from e

                result = response.json()

                if "results" not in result:
                    raise RuntimeError("vLLM rerank API response missing 'results' field")

                rerank_data = [convert_result_item(item) for item in result.get("results")]
                rerank_data.sort(key=lambda entry: entry.relevance_score, reverse=True)

                return RerankResponse(data=rerank_data)

        except httpx.HTTPError as e:
            raise ConnectionError(f"Failed to connect to vLLM rerank API at {endpoint}: {e}") from e
