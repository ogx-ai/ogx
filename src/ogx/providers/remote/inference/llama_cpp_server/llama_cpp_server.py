# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import httpx

from ogx.log import get_logger
from ogx.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from ogx.providers.utils.inference.models_dev_registry import classify_model
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx.providers.utils.inference.server_signature import (
    ServerUnreachableError,
    check_llama_cpp_server,
    verify_llama_cpp_server,
)
from ogx_api import (
    HealthResponse,
    Model,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    RerankData,
    RerankResponse,
)
from ogx_api.inference import RerankRequest

logger = get_logger(name=__name__, category="inference::llama_cpp_server")


class LlamaCppServerInferenceAdapter(OpenAIMixin):
    """Inference adapter for llama.cpp servers with an OpenAI-compatible API."""

    config: LlamaCppServerConfig

    def get_api_key(self) -> str | None:
        if self.config.auth_credential is None:
            return "NO KEY REQUIRED"
        return self.config.auth_credential.get_secret_value()

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    def _signature_api_key(self) -> str | None:
        return self.config.auth_credential.get_secret_value() if self.config.auth_credential else None

    async def initialize(self) -> None:
        # Fail fast at construction time if the server is running but is not a
        # llama.cpp server. A server that is simply not up yet only warns,
        # matching the Ollama adapter's behaviour.
        try:
            await verify_llama_cpp_server(self.get_base_url(), api_key=self._signature_api_key())
        except ServerUnreachableError as e:
            logger.warning(
                "llama.cpp server is not running; it must be reachable before models can be listed",
                base_url=self.get_base_url(),
                error=str(e),
            )

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying the server identifies as llama.cpp
        via its native GET /props endpoint.
        This method is used by the Provider API to verify that the service is
        running correctly.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        return await check_llama_cpp_server(self.get_base_url(), api_key=self._signature_api_key())

    def construct_model_from_identifier(self, identifier: str) -> Model:
        # llama.cpp's /v1/models response does not expose a model task/type field
        # (a router can serve llm and embedding models on one endpoint), so we
        # classify with models.dev with a name fallback, mirroring the vLLM adapter.
        # classify_model() covers both embedding and rerank classification.
        model = classify_model(identifier, self.__provider_id__)
        if model is not None:
            return model
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
                raise ValueError("llama.cpp rerank API does not support images")
            else:
                raise ValueError("Unsupported item type for reranking")

        payload: dict[str, str | int | float | list[str]] = {
            "model": request.model,
            "query": format_item(request.query),
            "documents": [format_item(item) for item in request.items],
        }
        if request.max_num_results is not None:
            payload["top_n"] = request.max_num_results

        # config.base_url already includes the /v1 prefix, and llama.cpp exposes
        # the (Jina-compatible) rerank endpoint at /v1/rerank.
        endpoint = self.get_base_url() + "/rerank"

        headers: dict[str, str] = {}
        api_key = self._get_api_key_from_config_or_provider_data()
        if api_key and api_key != "NO KEY REQUIRED":
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(**self._build_httpx_client_kwargs()) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"llama.cpp rerank API request failed with status {response.status_code}: {response.text}"
                    )

                def convert_result_item(item: dict) -> RerankData:
                    if "index" not in item or "relevance_score" not in item:
                        raise RuntimeError(
                            "llama.cpp rerank API response missing required fields 'index' or 'relevance_score'"
                        )

                    try:
                        return RerankData(index=int(item["index"]), relevance_score=float(item["relevance_score"]))
                    except (TypeError, ValueError) as e:
                        raise RuntimeError(f"Invalid data types in llama.cpp rerank API response: {e}") from e

                result = response.json()

                if "results" not in result:
                    raise RuntimeError("llama.cpp rerank API response missing 'results' field")

                rerank_data = [convert_result_item(item) for item in result.get("results")]
                rerank_data.sort(key=lambda entry: entry.relevance_score, reverse=True)

                return RerankResponse(data=rerank_data)

        except httpx.HTTPError as e:
            raise ConnectionError(f"Failed to connect to llama.cpp rerank API at {endpoint}: {e}") from e
