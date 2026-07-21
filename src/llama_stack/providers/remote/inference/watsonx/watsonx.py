# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
import requests
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from llama_stack.log import get_logger
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack_api import (
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

# IAM token cache
_iam_token: str | None = None
_iam_token_expiry: float = 0


def _get_iam_token(api_key: str) -> str:
    """Exchange a WatsonX API key for an IAM bearer token, with caching.

    WatsonX does not accept API keys directly for authentication. The AsyncOpenAI
    client sends the key as `Authorization: Bearer <token>`, but WatsonX requires
    an IAM token obtained by exchanging the API key with IBM's IAM service.
    Previously LiteLLM handled this internally; with the direct OpenAI mixin we
    perform the exchange ourselves.
    """
    global _iam_token, _iam_token_expiry

    # Return cached token if still valid (with 60s buffer)
    if _iam_token and time.time() < _iam_token_expiry - 60:
        return _iam_token

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token = data["access_token"]
    _iam_token_expiry = data.get("expiration", time.time() + 3600)
    return _iam_token


class WatsonXInferenceAdapter(OpenAIMixin):
    _model_cache: dict[str, Model] = {}

    provider_data_api_key_field: str = "watsonx_api_key"

    # WatsonX does not support stream_options
    supports_stream_options: bool = False

    def __init__(self, config: WatsonXConfig):
        super().__init__(config=config)

    def get_base_url(self) -> str:
        return f"{str(self.config.base_url).rstrip('/')}/ml/v1"

    def get_extra_client_params(self) -> dict[str, Any]:
        return {
            "default_query": {"version": WATSONX_API_VERSION},
            "timeout": httpx.Timeout(self.config.timeout),
        }

    @property
    def client(self) -> AsyncOpenAI:
        # Get the API key from config or provider data headers
        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            raise ValueError(
                "WatsonX API key not provided. Set WATSONX_API_KEY or pass it via "
                f'x-llamastack-provider-data: {{"{self.provider_data_api_key_field}": "<API_KEY>"}}'
            )

        # Exchange for IAM token before creating client
        iam_token = _get_iam_token(api_key)

        extra_params = self.get_extra_client_params()
        extra_params["http_client"] = DefaultAsyncHttpxClient(verify=self.shared_ssl_context)

        return AsyncOpenAI(
            api_key=iam_token,
            base_url=self.get_base_url(),
            **extra_params,
        )

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def list_provider_model_ids(self) -> Iterable[str]:
        """List models using WatsonX's /v1/models which requires project_id as query param."""
        client = self.client
        async with client:
            model_ids = [m.id async for m in client.models.list(extra_query={"project_id": self.config.project_id})]
        return model_ids

    def construct_model_from_identifier(self, identifier: str) -> Model:
        """Construct model with proper type based on identifier."""
        model_type = ModelType.llm
        metadata: dict[str, Any] = {}

        for spec in self._get_model_specs():
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
        # Reconstruct with extra fields so Pydantic picks them up in model_extra
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
        params = self._inject_project_id(params)
        return await super().openai_embeddings(params)

    def _get_model_specs(self) -> list[dict[str, Any]]:
        """Retrieves foundation model specifications from the watsonx.ai API."""
        url = f"{str(self.config.base_url)}/ml/v1/foundation_model_specs?version={WATSONX_API_VERSION}"
        response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "resources" not in data:
            raise ValueError("Resources not found in response")
        return data["resources"]
