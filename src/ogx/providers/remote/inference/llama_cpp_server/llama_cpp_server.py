# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import re

import httpx2

from ogx.log import get_logger
from ogx.providers.remote.inference.llama_cpp_server.config import LlamaCppServerConfig
from ogx.providers.utils.inference.models_dev_registry import classify_model
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx.providers.utils.inference.server_signature import (
    ServerUnreachableError,
    check_llama_cpp_server,
    verify_llama_cpp_server,
)
from ogx.providers.utils.inference.systemone import LOCAL_ENGINE_API_KEY, systemone_judge
from ogx_api import (
    HealthResponse,
    Model,
    ModelType,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    RerankData,
    RerankResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from ogx_api.inference import RerankRequest

logger = get_logger(name=__name__, category="inference::llama_cpp_server")

# Names of decision models available as GGUF for llama.cpp, which serves the SystemOne
# (decision) protocol natively since v0.6.0. llama.cpp's /v1/models listing does not
# expose per-model capabilities, so decision models are recognized by name.
_SYSTEMONE_MODEL_NAMES = frozenset({"laya", "julia-1", "openjev", "lev", "kev", "pplx-decider", "lfm2-d1"})

# llama.cpp derives model IDs from GGUF filenames (e.g. "Laya-Q8_0.gguf"), so strip a
# trailing quant suffix before matching.
_QUANT_SUFFIX = re.compile(r"-(?:q\d+(?:_\d+)?(?:_k_[a-z]+)?|bf16|f16|f32)$", re.IGNORECASE)


def _is_systemone_model(identifier: str) -> bool:
    name = identifier.split(":", 1)[0]
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    name = _QUANT_SUFFIX.sub("", name)
    return name.casefold() in _SYSTEMONE_MODEL_NAMES


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
            await verify_llama_cpp_server(
                self.get_base_url(), api_key=self._signature_api_key(), client_kwargs=self._build_httpx_client_kwargs()
            )
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
        return await check_llama_cpp_server(
            self.get_base_url(), api_key=self._signature_api_key(), client_kwargs=self._build_httpx_client_kwargs()
        )

    def construct_model_from_identifier(self, identifier: str) -> Model:
        # llama.cpp's /v1/models response does not expose a model task/type field
        # (a router can serve llm and embedding models on one endpoint), so we
        # classify with models.dev with a name fallback, mirroring the vLLM adapter.
        # classify_model() covers both embedding and rerank classification.
        model = classify_model(identifier, self.__provider_id__)
        if model is None:
            model = super().construct_model_from_identifier(identifier)
        if _is_systemone_model(identifier):
            # Decision models answer SystemOne judgments, not chat completions; register them
            # under ModelType.systemone so the /v1/systemone router finds them.
            model.model_type = ModelType.systemone
        return model

    def _get_systemone_base_url(self) -> str:
        # The SystemOne endpoint lives at the server root, but config.base_url carries the
        # /v1 prefix used by the OpenAI-compatible paths.
        base_url_str = self.get_base_url()
        if base_url_str.endswith("/v1"):
            return base_url_str[:-3]
        return base_url_str

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        """Answer a SystemOne judgment request by forwarding it to llama.cpp's /v1/systemone.

        llama.cpp serves the TypeSafe-compatible SystemOne protocol natively for local decision
        models, so no translation is needed; the shared helper validates the raw response body
        into the OGX wire model.
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
            self._get_systemone_base_url(),
            api_key,
            request,
            network=self.config.network,
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
            async with httpx2.AsyncClient(**self._build_httpx_client_kwargs()) as client:
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

        except httpx2.HTTPError as e:
            raise ConnectionError(f"Failed to connect to llama.cpp rerank API at {endpoint}: {e}") from e
