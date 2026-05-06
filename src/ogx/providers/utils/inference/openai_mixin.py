# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import ssl
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, ConfigDict, Field

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx.providers.utils.inference.http_client import (
    _merge_network_config_into_client,
    build_network_client_kwargs,
)
from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx.providers.utils.inference.openai_compat import (
    get_stream_options_for_telemetry,
    prepare_openai_completion_params,
)
from ogx.providers.utils.inference.prompt_adapter import localize_image_content
from ogx_api import (
    Model,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    OpenAIMessageParam,
    validate_embeddings_input_is_text,
)

logger = get_logger(name=__name__, category="providers::utils")


class OpenAIMixin(NeedsRequestProviderData, ABC, BaseModel):
    """
    Mixin class that provides OpenAI-specific functionality for inference providers.
    This class handles direct OpenAI API calls using the AsyncOpenAI client.

    This is an abstract base class that requires child classes to implement:
    - get_base_url(): Method to retrieve the OpenAI-compatible API base URL

    The behavior of this class can be customized by child classes in the following ways:
    - overwrite_completion_id: If True, overwrites the 'id' field in OpenAI responses
    - download_images: If True, downloads images and converts to base64 for providers that require it
    - supports_stream_options: If False, disables stream_options injection for providers that don't support it
    - embedding_model_metadata: A dictionary mapping model IDs to their embedding metadata
    - construct_model_from_identifier: Method to construct a Model instance corresponding to the given identifier
    - provider_data_api_key_field: Optional field name in provider data to look for API key
    - list_provider_model_ids: Method to list available models from the provider
    - get_extra_client_params: Method to provide extra parameters to the AsyncOpenAI client

    Expected Dependencies:
    - self.model_store: Injected by the OGX distribution system at runtime.
      This provides model registry functionality for looking up registered models.
      The model_store is set in routing_tables/common.py during provider initialization.
    """

    # Allow extra fields so the routing infra can inject model_store, __provider_id__, etc.
    # Allow arbitrary types for shared_ssl_context
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    config: RemoteInferenceProviderConfig

    # Allow subclasses to control whether to overwrite the 'id' field in OpenAI responses
    # is overwritten with a client-side generated id.
    #
    # This is useful for providers that do not return a unique id in the response.
    overwrite_completion_id: bool = False

    # Allow subclasses to control whether to download images and convert to base64
    # for providers that require base64 encoded images instead of URLs.
    download_images: bool = False

    # Allow subclasses to control whether the provider supports stream_options parameter
    # Set to False for providers that don't support stream_options (e.g., Ollama, vLLM)
    supports_stream_options: bool = True

    # Allow subclasses to control whether the provider supports tokenized embeddings input
    # Set to True for providers that support pre-tokenized input (list[int] and list[list[int]])
    supports_tokenized_embeddings_input: bool = False

    # Embedding model metadata for this provider
    # Can be set by subclasses or instances to provide embedding models
    # Format: {"model_id": {"embedding_dimension": 1536, "context_length": 8192}}
    embedding_model_metadata: dict[str, dict[str, int]] = {}

    # Cache of available models keyed by model ID
    # This is set in list_models() and used in check_model_availability()
    _model_cache: dict[str, Model] = {}

    # Optional field name in provider data to look for API key, which takes precedence
    provider_data_api_key_field: str | None = None

    # Shared SSL context for all calls to improve performance
    # SSL context construction touches disk and is expensive
    # Trade-off: SSL context changes require server restart
    shared_ssl_context: ssl.SSLContext | bool = Field(default_factory=ssl.create_default_context, exclude=True)

    def get_api_key(self) -> str | None:
        """
        Get the API key.

        :return: The API key as a string, or None if not set
        """
        if self.config.auth_credential is None:
            return None
        return self.config.auth_credential.get_secret_value()

    @abstractmethod
    def get_base_url(self) -> str:
        """
        Get the OpenAI-compatible API base URL.

        This method must be implemented by child classes to provide the base URL
        for the OpenAI API or compatible endpoints (e.g., "https://api.openai.com/v1").

        :return: The base URL as a string
        """
        pass

    def get_extra_client_params(self) -> dict[str, Any]:
        """
        Get any extra parameters to pass to the AsyncOpenAI client.

        Child classes can override this method to provide additional parameters
        such as custom http_client, timeout settings, proxies, etc.

        Note: Network configuration from config.network is automatically applied
        in the client property. This method is for provider-specific customizations.

        :return: A dictionary of extra parameters
        """
        return {}

    def construct_model_from_identifier(self, identifier: str) -> Model:
        """
        Construct a Model instance corresponding to the given identifier

        Child classes can override this to customize model typing/metadata.

        :param identifier: The provider's model identifier
        :return: A Model instance
        """
        if metadata := self.embedding_model_metadata.get(identifier):
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.embedding,
                metadata=metadata,
            )
        return Model(
            provider_id=self.__provider_id__,  # type: ignore[attr-defined]
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.llm,
        )

    async def list_provider_model_ids(self) -> Iterable[str]:
        """
        List available models from the provider.

        Child classes can override this method to provide a custom implementation
        for listing models. The default implementation uses the AsyncOpenAI client
        to list models from the OpenAI-compatible endpoint.

        :return: An iterable of model IDs or None if not implemented
        """
        client = self.client
        async with client:
            model_ids = [m.id async for m in client.models.list()]
        return model_ids

    async def initialize(self) -> None:
        """
        Initialize the OpenAI mixin.

        This method provides a default implementation that does nothing.
        Subclasses can override this method to perform initialization tasks
        such as setting up clients, validating configurations, etc.
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the OpenAI mixin.

        This method provides a default implementation that does nothing.
        Subclasses can override this method to perform cleanup tasks
        such as closing connections, releasing resources, etc.
        """
        pass

    @property
    def client(self) -> AsyncOpenAI:
        """
        Get an AsyncOpenAI client instance.

        Uses the abstract methods get_api_key() and get_base_url() which must be
        implemented by child classes.

        Network configuration from config.network is automatically applied.
        Users can also provide the API key via the provider data header, which
        is used instead of any config API key.
        """

        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            message = "API key not provided."
            if self.provider_data_api_key_field:
                message += f' Please provide a valid API key in the provider data header, e.g. x-ogx-provider-data: {{"{self.provider_data_api_key_field}": "<API_KEY>"}}.'
            raise ValueError(message)

        extra_params = self.get_extra_client_params()
        network_kwargs = build_network_client_kwargs(self.config.network)

        # Handle http_client creation/merging:
        # - If get_extra_client_params() provides an http_client (e.g., OCI with custom auth),
        #   merge network config into it. The merge behavior:
        #   * Preserves auth from get_extra_client_params() (provider-specific auth like OCI signer)
        #   * Preserves headers from get_extra_client_params() as base
        #   * Applies network config (TLS, proxy, timeout, headers) on top
        #   * Network config headers take precedence over provider headers (allows override)
        # - Otherwise, if network config exists, create http_client from it
        # - Otherwise, use a cached SSL context for performance
        # This allows providers with custom auth to still use standard network settings
        if "http_client" in extra_params:
            if network_kwargs:
                extra_params["http_client"] = _merge_network_config_into_client(
                    extra_params["http_client"], self.config.network
                )
        elif network_kwargs:
            extra_params["http_client"] = httpx.AsyncClient(**network_kwargs)
        else:
            extra_params["http_client"] = DefaultAsyncHttpxClient(verify=self.shared_ssl_context)

        return AsyncOpenAI(
            api_key=api_key,
            base_url=self.get_base_url(),
            **extra_params,
        )

    def _get_api_key_from_config_or_provider_data(self) -> str | None:
        api_key = self.get_api_key()

        if self.provider_data_api_key_field:
            provider_data = self.get_request_provider_data()
            if provider_data and getattr(provider_data, self.provider_data_api_key_field, None):
                value = getattr(provider_data, self.provider_data_api_key_field)
                api_key = value.get_secret_value()

        return api_key

    def _validate_model_allowed(self, provider_model_id: str) -> None:
        """
        Validate that the model is in the allowed_models list if configured.

        :param provider_model_id: The provider-specific model ID to validate
        :raises ValueError: If the model is not in the allowed_models list
        """
        if self.config.allowed_models is not None and provider_model_id not in self.config.allowed_models:
            raise ValueError(
                f"Model '{provider_model_id}' is not in the allowed models list. "
                f"Allowed models: {self.config.allowed_models}"
            )

    async def _get_provider_model_id(self, model: str) -> str:
        """
        Get the provider-specific model ID from the model store.

        This is a utility method that looks up the registered model and returns
        the provider_resource_id that should be used for actual API calls.

        :param model: The registered model name/identifier
        :return: The provider-specific model ID (e.g., "gpt-4")
        """
        # self.model_store is injected by the distribution system at runtime
        if not await self.model_store.has_model(model):  # type: ignore[attr-defined]
            return model

        # Look up the registered model to get the provider-specific model ID
        model_obj: Model = await self.model_store.get_model(model)  # type: ignore[attr-defined]
        # provider_resource_id is str | None, but we expect it to be str for OpenAI calls
        if model_obj.provider_resource_id is None:
            raise ValueError(f"Model {model} has no provider_resource_id")
        return model_obj.provider_resource_id

    async def _maybe_overwrite_id(self, resp: Any, stream: bool | None) -> Any:
        if stream:
            new_id = f"cltsd-{uuid.uuid4()}" if self.overwrite_completion_id else None

            async def _gen():
                async for chunk in resp:
                    if new_id:
                        chunk.id = new_id
                    yield chunk

            return _gen()
        else:
            if self.overwrite_completion_id:
                resp.id = f"cltsd-{uuid.uuid4()}"
            return resp

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """
        Direct OpenAI completion API call.
        """
        # Inject stream_options when streaming and telemetry is active
        stream_options = get_stream_options_for_telemetry(
            params.stream_options, params.stream or False, self.supports_stream_options
        )

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)

        completion_kwargs = await prepare_openai_completion_params(
            model=provider_model_id,
            prompt=params.prompt,
            best_of=params.best_of,
            echo=params.echo,
            frequency_penalty=params.frequency_penalty,
            logit_bias=params.logit_bias,
            logprobs=params.logprobs,
            max_tokens=params.max_tokens,
            n=params.n,
            presence_penalty=params.presence_penalty,
            seed=params.seed,
            stop=params.stop,
            stream=params.stream,
            stream_options=stream_options,
            temperature=params.temperature,
            top_p=params.top_p,
            user=params.user,
            suffix=params.suffix,
        )
        if extra_body := params.model_extra:
            completion_kwargs["extra_body"] = extra_body
        resp = await self.client.completions.create(**completion_kwargs)

        return await self._maybe_overwrite_id(resp, params.stream)  # type: ignore[no-any-return]

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """
        Direct OpenAI chat completion API call.
        """
        # Inject stream_options when streaming and telemetry is active
        stream_options = get_stream_options_for_telemetry(
            params.stream_options, params.stream or False, self.supports_stream_options
        )

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)

        messages = params.messages

        if self.download_images:

            async def _localize_image_url(m: OpenAIMessageParam) -> OpenAIMessageParam:
                if isinstance(m.content, list):
                    for c in m.content:
                        if c.type == "image_url" and c.image_url and c.image_url.url and "http" in c.image_url.url:
                            localize_result = await localize_image_content(c.image_url.url)
                            if localize_result is None:
                                raise ValueError(
                                    f"Failed to localize image content from {c.image_url.url[:42]}{'...' if len(c.image_url.url) > 42 else ''}"
                                )
                            content, format = localize_result
                            c.image_url.url = f"data:image/{format};base64,{base64.b64encode(content).decode('utf-8')}"
                # else it's a string and we don't need to modify it
                return m

            messages = [await _localize_image_url(m) for m in messages]

        request_params = await prepare_openai_completion_params(
            model=provider_model_id,
            messages=messages,
            frequency_penalty=params.frequency_penalty,
            function_call=params.function_call,
            functions=params.functions,
            logit_bias=params.logit_bias,
            logprobs=params.logprobs,
            max_completion_tokens=params.max_completion_tokens,
            max_tokens=params.max_tokens,
            n=params.n,
            parallel_tool_calls=params.parallel_tool_calls,
            presence_penalty=params.presence_penalty,
            response_format=params.response_format,
            seed=params.seed,
            stop=params.stop,
            stream=params.stream,
            stream_options=stream_options,
            temperature=params.temperature,
            tool_choice=params.tool_choice,
            tools=params.tools,
            top_logprobs=params.top_logprobs,
            top_p=params.top_p,
            user=params.user,
            safety_identifier=params.safety_identifier,
            service_tier=params.service_tier,
            reasoning_effort=params.reasoning_effort,
            prompt_cache_key=params.prompt_cache_key,
        )

        if extra_body := params.model_extra:
            request_params["extra_body"] = extra_body
        resp = await self.client.chat.completions.create(**request_params)

        return await self._maybe_overwrite_id(resp, params.stream)  # type: ignore[no-any-return]

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """
        Direct OpenAI embeddings API call.
        """
        # Validate token array support if provider doesn't support it
        if not self.supports_tokenized_embeddings_input:
            validate_embeddings_input_is_text(params)

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)

        # Build request params conditionally to avoid NotGiven/Omit type mismatch
        # The OpenAI SDK uses Omit in signatures but NOT_GIVEN has type NotGiven
        request_params: dict[str, Any] = {
            "model": provider_model_id,
            "input": params.input,
        }
        if params.encoding_format is not None:
            request_params["encoding_format"] = params.encoding_format
        if params.dimensions is not None:
            request_params["dimensions"] = params.dimensions
        if params.user is not None:
            request_params["user"] = params.user
        if params.model_extra:
            request_params["extra_body"] = params.model_extra

        response = await self.client.embeddings.create(**request_params)

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=params.model,
            usage=usage,
        )

    ###
    # ModelsProtocolPrivate implementation - provide model management functionality
    #
    #  async def register_model(self, model: Model) -> Model: ...
    #  async def unregister_model(self, model_id: str) -> None: ...
    #
    #  async def list_models(self) -> list[Model] | None: ...
    #  async def should_refresh_models(self) -> bool: ...
    ##

    async def register_model(self, model: Model) -> Model:
        # Check if we should validate model availability (defaults to False)
        should_validate = bool(model.model_validation)

        if not should_validate:
            logger.debug(
                "Skipping model availability check for (model_validation=false)",
                provider_model_id=model.provider_model_id,
            )
            return model

        if not await self.check_model_availability(model.provider_model_id):
            raise ValueError(f"Model {model.provider_model_id} is not available from provider {self.__provider_id__}")  # type: ignore[attr-defined]
        return model

    async def unregister_model(self, model_id: str) -> None:
        return None

    async def list_models(self) -> list[Model] | None:
        """
        List available models from the provider's /v1/models endpoint augmented with static embedding model metadata.

        Also, caches the models in self._model_cache for use in check_model_availability().

        :return: A list of Model instances representing available models.
        """
        self._model_cache = {}

        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            logger.debug(
                "list_provider_model_ids() disabled because API key not provided", provider=self.__class__.__name__
            )
            return None

        try:
            iterable = await self.list_provider_model_ids()
        except Exception as e:
            logger.error("list_provider_model_ids() failed", provider=self.__class__.__name__, error=str(e))
            raise
        if not hasattr(iterable, "__iter__"):
            raise TypeError(
                f"Failed to list models: {self.__class__.__name__}.list_provider_model_ids() must return an iterable of "
                f"strings, but returned {type(iterable).__name__}"
            )

        provider_models_ids = list(iterable)
        logger.info(
            "list_provider_model_ids() returned models",
            provider=self.__class__.__name__,
            count=len(provider_models_ids),
        )

        for provider_model_id in provider_models_ids:
            if not isinstance(provider_model_id, str):
                raise ValueError(f"Model ID {provider_model_id} from list_provider_model_ids() is not a string")
            if self.config.allowed_models is not None and provider_model_id not in self.config.allowed_models:
                logger.info("Skipping model not in allowed models list", model=provider_model_id)
                continue
            model = self.construct_model_from_identifier(provider_model_id)
            self._model_cache[provider_model_id] = model

        return list(self._model_cache.values())

    async def check_model_availability(self, model: str) -> bool:
        """
        Check if a specific model is available from the provider's /v1/models or pre-registered.

        :param model: The model identifier to check.
        :return: True if the model is available dynamically or pre-registered, False otherwise.
        """
        # First check if the model is pre-registered in the model store
        if hasattr(self, "model_store") and self.model_store:
            qualified_model = f"{self.__provider_id__}/{model}"  # type: ignore[attr-defined]
            if await self.model_store.has_model(qualified_model):
                return True

        # Then check the provider's dynamic model cache
        if not self._model_cache:
            await self.list_models()
        return model in self._model_cache

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

    #
    # The model_dump implementations are to avoid serializing the extra fields,
    # e.g. model_store, which are not pydantic.
    #

    def _filter_fields(self, **kwargs):
        """Helper to exclude extra fields from serialization."""
        # Exclude any extra fields stored in __pydantic_extra__
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            exclude = kwargs.get("exclude", set())
            if not isinstance(exclude, set):
                exclude = set(exclude) if exclude else set()
            exclude.update(self.__pydantic_extra__.keys())
            kwargs["exclude"] = exclude
        return kwargs

    def model_dump(self, **kwargs):
        """Override to exclude extra fields from serialization."""
        kwargs = self._filter_fields(**kwargs)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs):
        """Override to exclude extra fields from JSON serialization."""
        kwargs = self._filter_fields(**kwargs)
        return super().model_dump_json(**kwargs)
