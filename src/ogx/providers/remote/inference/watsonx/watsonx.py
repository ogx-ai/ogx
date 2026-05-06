# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from ogx.log import get_logger
from ogx.providers.remote.inference.watsonx.config import WatsonXConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    Model,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    validate_embeddings_input_is_text,
)

logger = get_logger(name=__name__, category="providers::remote::watsonx")

WATSONX_API_VERSION = "2023-10-25"


class WatsonXInferenceAdapter(OpenAIMixin):
    """Inference adapter for IBM WatsonX AI platform."""

    _model_cache: dict[str, Model] = {}

    provider_data_api_key_field: str = "watsonx_api_key"

    # WatsonX does not support stream_options
    supports_stream_options: bool = False

    def __init__(self, config: WatsonXConfig):
        super().__init__(config=config)
        self._iam_token_cache: dict[str, tuple[str, float]] = {}
        self._model_specs_cache: list[dict[str, Any]] | None = None

    def get_base_url(self) -> str:
        return f"{str(self.config.base_url).rstrip('/')}/ml/v1"

    def get_extra_client_params(self) -> dict[str, Any]:
        return {
            "default_query": {"version": WATSONX_API_VERSION},
            "timeout": httpx.Timeout(self.config.timeout),
        }

    async def _refresh_iam_token(self, api_key: str) -> str:
        """Exchange a WatsonX API key for an IAM bearer token asynchronously.

        WatsonX does not accept API keys directly for authentication. The AsyncOpenAI
        client sends the key as ``Authorization: Bearer <token>``, but WatsonX requires
        an IAM token obtained by exchanging the API key with IBM's IAM service.

        The token cache is keyed by the API key so that per-request credentials
        (passed via provider_data) do not leak across requests.

        Returns the cached token if still valid (with 60 s buffer).
        """
        cached = self._iam_token_cache.get(api_key)
        if cached:
            token, expiry = cached
            if time.time() < expiry - 60:
                return token

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(
                    "https://iam.cloud.ibm.com/identity/token",
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    content=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                token = data["access_token"]
                expiry = data.get("expiration", time.time() + 3600)
                self._iam_token_cache[api_key] = (token, expiry)
                return token
        except Exception as e:
            logger.warning("IAM token exchange failed, using API key directly", error=str(e))
            return api_key

    def _get_api_key_or_raise(self) -> str:
        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            raise ValueError(
                "WatsonX API key not provided. Set WATSONX_API_KEY or pass it via "
                f'x-ogx-provider-data: {{"{self.provider_data_api_key_field}": "<API_KEY>"}}'
            )
        return api_key

    async def _ensure_client(self) -> AsyncOpenAI:
        """Build an AsyncOpenAI client with a fresh IAM token (non-blocking)."""
        api_key = self._get_api_key_or_raise()
        iam_token = await self._refresh_iam_token(api_key)

        extra_params = self.get_extra_client_params()
        extra_params["http_client"] = DefaultAsyncHttpxClient(verify=self.shared_ssl_context)

        return AsyncOpenAI(
            api_key=iam_token,
            base_url=self.get_base_url(),
            **extra_params,
        )

    @property
    def client(self) -> AsyncOpenAI:
        # Fast path: reuse cached IAM token (no network I/O).
        # _ensure_client() must be awaited before the first call to pre-populate the token.
        api_key = self._get_api_key_or_raise()
        cached = self._iam_token_cache.get(api_key)
        iam_token = cached[0] if cached and time.time() < cached[1] - 60 else api_key

        extra_params = self.get_extra_client_params()
        extra_params["http_client"] = DefaultAsyncHttpxClient(verify=self.shared_ssl_context)

        return AsyncOpenAI(
            api_key=iam_token,
            base_url=self.get_base_url(),
            **extra_params,
        )

    async def initialize(self) -> None:
        self._model_specs_cache = await self._fetch_model_specs()

    async def shutdown(self) -> None:
        pass

    async def list_provider_model_ids(self) -> Iterable[str]:
        """List models using WatsonX's /v1/models which requires project_id as query param."""
        client = await self._ensure_client()
        async with client:
            model_ids = [m.id async for m in client.models.list(extra_query={"project_id": self.config.project_id})]
        return model_ids

    def construct_model_from_identifier(self, identifier: str) -> Model:
        """Construct model with proper type based on identifier."""
        model_type = ModelType.llm
        metadata: dict[str, Any] = {}

        specs = self._model_specs_cache or []
        for spec in specs:
            if spec["model_id"] == identifier:
                functions = [f["id"] for f in spec.get("functions", [])]
                if "embedding" in functions:
                    model_type = ModelType.embedding
                    metadata = {
                        "embedding_dimension": spec.get("model_limits", {}).get("embedding_dimension", 0),
                        "context_length": spec.get("model_limits", {}).get("max_sequence_length", 0),
                    }
                break

        return Model(
            provider_id=self.__provider_id__,
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=model_type,
            metadata=metadata,
        )

    def _inject_project_id(self, params: Any) -> Any:
        """Inject project_id into model_extra so it's sent as extra_body to the API."""
        extra = dict(params.model_extra) if params.model_extra else {}
        extra["project_id"] = self.config.project_id
        data = params.model_dump()
        data.update(extra)
        return type(params)(**data)

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        # Strip parallel_tool_calls — WatsonX doesn't support it
        if params.parallel_tool_calls is not None:
            params = params.model_copy(update={"parallel_tool_calls": None})

        await self._refresh_iam_token(self._get_api_key_or_raise())
        params = self._inject_project_id(params)
        return await super().openai_chat_completion(params)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        raise NotImplementedError(
            "WatsonX does not support the /v1/completions endpoint. Use /v1/chat/completions instead."
        )

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        validate_embeddings_input_is_text(params)
        await self._refresh_iam_token(self._get_api_key_or_raise())
        params = self._inject_project_id(params)
        return await super().openai_embeddings(params)

    async def _fetch_model_specs(self) -> list[dict[str, Any]]:
        """Retrieve foundation model specifications from the WatsonX API."""
        url = f"{str(self.config.base_url)}/ml/v1/foundation_model_specs?version={WATSONX_API_VERSION}"
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(url, headers={"Content-Type": "application/json"}, timeout=30)
            response.raise_for_status()
            data = response.json()
        if "resources" not in data:
            raise ValueError("Resources not found in response")
        return data["resources"]
