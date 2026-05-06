# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import base64
import struct
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from google.genai import Client
from google.genai import types as genai_types
from google.oauth2.credentials import Credentials
from pydantic import BaseModel, ConfigDict, PrivateAttr

from ogx.core.request_headers import NeedsRequestProviderData
from ogx.log import get_logger
from ogx.providers.remote.inference.vertexai import converters
from ogx.providers.remote.inference.vertexai.config import (
    VertexAIConfig,
    VertexAIProviderDataValidator,
)
from ogx.providers.remote.inference.vertexai.utils import build_http_options as _build_http_options
from ogx.providers.utils.inference.openai_compat import get_stream_options_for_telemetry
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
    RerankResponse,
    validate_embeddings_input_is_text,
)
from ogx_api.inference import RerankRequest

logger = get_logger(__name__, category="inference")


class GeminiSamplingParams(BaseModel):
    """Gemini sampling parameters mapped from OpenAI request fields.

    Field names follow Gemini conventions so that ``model_dump(exclude_none=True)``
    produces kwargs ready for ``GenerateContentConfig`` with no manual if-not-None
    checks.  Use ``from_openai_params()`` to construct from an OpenAI request.
    """

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    candidate_count: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    # Gemini swaps logprobs semantics vs OpenAI:
    # OpenAI logprobs (bool) → Gemini response_logprobs (bool)
    # OpenAI top_logprobs (int) → Gemini logprobs (int count)
    response_logprobs: bool | None = None
    logprobs: int | None = None

    @classmethod
    def from_openai_params(cls, params: OpenAIChatCompletionRequestWithExtraBody) -> GeminiSamplingParams:
        """Create from OpenAI request parameters, resolving field priorities and types.

        Handles two transform cases:
        - ``max_completion_tokens`` takes priority over ``max_tokens``
        - ``stop`` (str | list) is normalized to a list
        """
        max_tokens = params.max_completion_tokens if params.max_completion_tokens is not None else params.max_tokens

        stop_sequences: list[str] | None = None
        if params.stop is not None:
            stop_sequences = [params.stop] if isinstance(params.stop, str) else list(params.stop)

        return cls(
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            seed=params.seed,
            candidate_count=params.n,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences,
            response_logprobs=params.logprobs,
            logprobs=params.top_logprobs,
        )


class GeminiCompletionSamplingParams(BaseModel):
    """Gemini sampling parameters mapped from OpenAI text completion request fields.

    Mirrors ``GeminiSamplingParams`` for the ``/v1/completions`` endpoint.
    ``model_dump(exclude_none=True)`` produces kwargs ready for
    ``GenerateContentConfig``.
    """

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    candidate_count: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    response_logprobs: bool | None = None

    @classmethod
    def from_openai_params(cls, params: OpenAICompletionRequestWithExtraBody) -> GeminiCompletionSamplingParams:
        """Create from OpenAI text completion request parameters."""
        stop_sequences: list[str] | None = None
        if params.stop is not None:
            stop_sequences = [params.stop] if isinstance(params.stop, str) else list(params.stop)

        return cls(
            temperature=params.temperature,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty,
            seed=params.seed,
            candidate_count=params.n,
            max_output_tokens=params.max_tokens,
            stop_sequences=stop_sequences,
            response_logprobs=params.logprobs or None,
        )


class VertexAIInferenceAdapter(NeedsRequestProviderData, BaseModel):
    """Inference adapter for Google Vertex AI platform."""

    # extra="allow" lets the routing infra inject model_store, __provider_id__, etc.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    config: VertexAIConfig
    _default_client: Client | None = PrivateAttr(default=None)
    _http_options: genai_types.HttpOptions | None = PrivateAttr(default=None)
    _http_options_initialized: bool = PrivateAttr(default=False)
    _model_cache: dict[str, Model] = PrivateAttr(default_factory=dict)
    embedding_model_metadata: dict[str, dict[str, int]] = {
        "publishers/google/models/text-embedding-004": {"embedding_dimension": 768, "context_length": 2048},
        "publishers/google/models/gemini-embedding-001": {"embedding_dimension": 3072, "context_length": 2048},
        # Gemini API format (vertexai=False) uses the "models/" prefix.
        "models/text-embedding-004": {"embedding_dimension": 768, "context_length": 2048},
        "models/gemini-embedding-001": {"embedding_dimension": 3072, "context_length": 2048},
    }

    async def _close_managed_httpx_client(self) -> None:
        if self._http_options is None:
            return

        client = getattr(self._http_options, "httpx_async_client", None)
        if client is not None:
            await client.aclose()

    def _ensure_http_options(self) -> None:
        """Lazily initialize HTTP options in the current event loop.

        This defers httpx.AsyncClient creation from initialize() to first use,
        avoiding event loop mismatch when the server creates providers in a
        temporary event loop during startup.
        """
        if self._http_options_initialized:
            return

        self._http_options = _build_http_options(self.config.network)
        self._http_options_initialized = True

    async def initialize(self) -> None:
        """Initialize the provider without creating httpx clients.

        HTTP options (including httpx.AsyncClient) are created lazily on first
        use to avoid event loop mismatch issues when initialization happens in
        a temporary event loop during server startup.

        We don't create the default client here because that would trigger
        _ensure_http_options() and create httpx.AsyncClient in the wrong event loop.
        The client will be created on first use via _get_client().
        """
        try:
            # Don't create the client here - it will be created lazily on first use
            # This avoids calling _ensure_http_options() in the temporary startup event loop
            logger.info(
                "VertexAI provider initialized for project=%s location=%s (client will be created on first use)",
                self.config.project,
                self.config.location,
            )
        except Exception:
            logger.warning(
                "Failed to initialize VertexAI provider configuration.",
                exc_info=True,
            )

    async def shutdown(self) -> None:
        await self._close_managed_httpx_client()
        self._http_options = None
        self._http_options_initialized = False
        self._default_client = None

    async def register_model(self, model: Model) -> Model:
        provider_resource_id = model.provider_resource_id or model.identifier
        if not await self.check_model_availability(provider_resource_id):
            raise ValueError(
                f"Model {provider_resource_id} is not available from provider {self.__provider_id__}"  # type: ignore[attr-defined]
            )
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    def _create_adc_client(self, project: str, location: str) -> Client:
        """Create a client using Application Default Credentials.

        Ensures HTTP options are initialized in the current event loop before
        creating the client.
        """
        self._ensure_http_options()
        kwargs: dict[str, Any] = dict(vertexai=True, project=project, location=location)
        if self._http_options is not None:
            kwargs["http_options"] = self._http_options
        return Client(**kwargs)

    def _create_client(self, project: str, location: str, *, access_token: str | None = None) -> Client:
        """Create a VertexAI client.

        When *access_token* is provided the client is **not** cached because
        OAuth tokens are short-lived (~1 h) and caching would cause silent
        auth failures after expiry.  ADC clients are not cached either because
        network configuration is instance-specific and not hashable for
        use as an lru_cache key.

        Ensures HTTP options are initialized in the current event loop before
        creating the client.
        """
        self._ensure_http_options()
        if access_token:
            credentials = Credentials(token=access_token)
            kwargs: dict[str, Any] = dict(vertexai=True, project=project, location=location, credentials=credentials)
            if self._http_options is not None:
                kwargs["http_options"] = self._http_options
            return Client(**kwargs)
        return self._create_adc_client(project, location)

    async def check_model_availability(self, model: str) -> bool:
        """Check whether *model* is served by the configured VertexAI project.

        Uses a cache populated by list_models() to avoid repeated API calls.
        Falls back to accepting the model when the cache is empty and the API
        is unreachable so that configured models are not rejected during offline startup.
        """
        if self._model_cache:
            return model in self._model_cache
        try:
            await self.list_models()
        except Exception:
            logger.warning(
                "Failed to list VertexAI models for availability check; accepting model '%s' without validation.",
                model,
                exc_info=True,
            )
            return True
        return model in self._model_cache

    @staticmethod
    def _is_usable_model(model: Any) -> bool:
        """Check if model is usable (supports generateContent or embedContent)."""
        actions = getattr(model, "supported_actions", None)
        if actions is None and isinstance(model, dict):
            actions = model.get("supported_actions")
        actions = actions or []
        if not actions:
            return True
        return "generateContent" in actions or "embedContent" in actions

    def _get_request_provider_overrides(self) -> VertexAIProviderDataValidator | None:
        provider_data = self.get_request_provider_data()
        if provider_data is None:
            return None

        if isinstance(provider_data, VertexAIProviderDataValidator):
            return provider_data

        if isinstance(provider_data, dict):
            return VertexAIProviderDataValidator(**provider_data)

        try:
            return VertexAIProviderDataValidator.model_validate(provider_data)
        except Exception:
            logger.warning("Failed to parse VertexAI provider data, falling back to config defaults", exc_info=True)
            return None

    def _get_client(self) -> Client:
        overrides = self._get_request_provider_overrides()
        if overrides is not None:
            project = overrides.vertex_project or self.config.project
            location = overrides.vertex_location or self.config.location
            if overrides.vertex_access_token:
                return self._create_client(
                    project=project,
                    location=location,
                    access_token=overrides.vertex_access_token.get_secret_value(),
                )
            if overrides.vertex_project or overrides.vertex_location:
                access_token = self.config.auth_credential.get_secret_value() if self.config.auth_credential else None
                return self._create_client(project=project, location=location, access_token=access_token)

        # Lazily create the default client on first use
        if self._default_client is None:
            access_token = self.config.auth_credential.get_secret_value() if self.config.auth_credential else None
            try:
                self._default_client = self._create_client(
                    project=self.config.project,
                    location=self.config.location,
                    access_token=access_token,
                )
                logger.info(
                    "Created default VertexAI client on first use",
                    project=self.config.project,
                    location=self.config.location,
                )
            except Exception:
                logger.error(
                    "Failed to create default VertexAI client. Pass credentials via X-OGX-Provider-Data header.",
                    exc_info=True,
                )
                raise ValueError(
                    "Failed to create default Vertex AI client. "
                    "Pass Vertex AI access token in the header X-OGX-Provider-Data "
                    'as { "vertex_access_token": <your access token> }'
                ) from None
        return self._default_client

    async def _get_provider_model_id(self, model: str) -> str:
        # model_store is injected at runtime by the routing infra
        if hasattr(self, "model_store") and self.model_store and await self.model_store.has_model(model):  # type: ignore[attr-defined]
            model_obj: Model = await self.model_store.get_model(model)  # type: ignore[attr-defined]
            if model_obj.provider_resource_id is None:
                raise ValueError(f"Model {model} has no provider_resource_id")
            return model_obj.provider_resource_id

        return model

    async def list_provider_model_ids(self) -> list[str]:
        """List model IDs available from the configured Vertex AI project.

        Returns model names exactly as the ``google-genai`` SDK provides them
        (e.g. ``publishers/google/models/gemini-2.5-flash`` for Vertex AI,
        ``models/gemini-2.5-flash`` for the Gemini API).  No prefix stripping
        is applied because the SDK's internal ``t_model()`` normalizer already
        accepts all resource-name formats when making API calls.
        """
        client = self._get_client()
        config = genai_types.ListModelsConfig(query_base=True)
        result: list[str] = []

        async for model in await client.aio.models.list(config=config):
            if not self._is_usable_model(model):
                continue

            name = getattr(model, "name", "") or ""
            if not name:
                continue
            result.append(name)

        return list(dict.fromkeys(result))

    async def list_models(self) -> list[Model] | None:
        """List models available from the configured VertexAI project.

        Queries the Gemini API via ``list_provider_model_ids()`` and constructs
        ``Model`` objects, respecting ``allowed_models`` when configured.
        Populates ``_model_cache`` for use by ``check_model_availability()``.
        """
        try:
            provider_model_ids = await self.list_provider_model_ids()
        except Exception:
            logger.error(
                "%s.list_provider_model_ids() failed",
                self.__class__.__name__,
                exc_info=True,
            )
            raise

        self._model_cache = {}
        models: list[Model] = []
        for provider_model_id in provider_model_ids:
            if self.config.allowed_models is not None and not self._is_model_allowed(provider_model_id):
                continue
            if metadata := self.embedding_model_metadata.get(provider_model_id):
                model = Model(
                    provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                    provider_resource_id=provider_model_id,
                    identifier=provider_model_id,
                    model_type=ModelType.embedding,
                    metadata=metadata,
                )
            else:
                model = Model(
                    provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                    provider_resource_id=provider_model_id,
                    identifier=provider_model_id,
                    model_type=ModelType.llm,
                )
            self._model_cache[provider_model_id] = model
            models.append(model)

        logger.info("list_models() returned models", provider=self.__class__.__name__, count=len(models))
        return models

    async def should_refresh_models(self) -> bool:
        return self.config.refresh_models

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

    def _build_tool_config(self, tool_choice: str | dict[str, Any] | None) -> genai_types.ToolConfig | None:
        if tool_choice is None or tool_choice == "auto":
            return None

        if tool_choice == "none":
            return self._make_tool_config(mode="NONE")

        if tool_choice == "required":
            return self._make_tool_config(mode="ANY")

        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("function", {}).get("name")
            if function_name:
                return self._make_tool_config(mode="ANY", allowed_function_names=[function_name])

        return None

    @staticmethod
    def _make_tool_config(
        *,
        mode: str,
        allowed_function_names: list[str] | None = None,
    ) -> genai_types.ToolConfig:
        function_calling_kwargs: dict[str, Any] = {"mode": cast(Any, mode)}
        if allowed_function_names:
            function_calling_kwargs["allowed_function_names"] = allowed_function_names
        function_calling = genai_types.FunctionCallingConfig(**function_calling_kwargs)
        return genai_types.ToolConfig(function_calling_config=function_calling)

    @staticmethod
    def _collect_sampling_params(params: OpenAIChatCompletionRequestWithExtraBody) -> dict[str, Any]:
        """Collect sampling-related config kwargs from the request.

        Uses ``GeminiSamplingParams`` to map OpenAI field names to their Gemini
        equivalents, with ``model_dump(exclude_none=True)`` replacing manual
        if-not-None checks.  ``response_format`` is handled separately because
        it requires a converter that produces multiple output keys.
        """
        kwargs = GeminiSamplingParams.from_openai_params(params).model_dump(exclude_none=True)

        if params.response_format is not None:
            kwargs.update(converters.convert_response_format(params.response_format.model_dump(exclude_none=True)))

        return kwargs

    @staticmethod
    def _build_thinking_config(reasoning_effort: str | None) -> genai_types.ThinkingConfig | None:
        """Map OpenAI reasoning_effort to Gemini ThinkingConfig.

        OpenAI's reasoning_effort levels map to Gemini's thinking configuration:
        - "none"    → thinking_budget=0  (disables thinking entirely)
        - "minimal" → ThinkingLevel.MINIMAL
        - "low"     → ThinkingLevel.LOW
        - "medium"  → ThinkingLevel.MEDIUM
        - "high"    → ThinkingLevel.HIGH
        - "xhigh"   → ThinkingLevel.HIGH  (no Gemini equivalent; closest is HIGH)
        """
        if reasoning_effort is None:
            return None

        if reasoning_effort == "none":
            return genai_types.ThinkingConfig(thinking_budget=0)

        effort_to_thinking_level: dict[str, str] = {
            "minimal": "MINIMAL",
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
            # "xhigh" has no Gemini equivalent; map to HIGH (closest available level)
            "xhigh": "HIGH",
        }

        level = effort_to_thinking_level.get(reasoning_effort)
        if level is None:
            logger.warning("Unknown reasoning_effort value, ignoring", reasoning_effort=repr(reasoning_effort))
            return None

        return genai_types.ThinkingConfig(thinking_level=cast(Any, level))

    @staticmethod
    def _convert_service_tier(service_tier: str | None) -> str | None:
        """Map OpenAI service_tier to a Gemini ServiceTier string.

        The google-genai SDK accepts case-insensitive strings on
        ``GenerateContentConfig.service_tier``.  OpenAI and Gemini use
        different vocabulary for the same concept:

        - ``"auto"``     → ``None``        (omit; let the API decide)
        - ``"default"``  → ``"standard"``  (Gemini's default tier)
        - ``"flex"``     → ``"flex"``
        - ``"priority"`` → ``"priority"``

        Accepts ``ServiceTier`` (a ``StrEnum``) or plain strings.
        """
        if service_tier is None:
            return None

        _map: dict[str, str | None] = {
            "auto": None,
            "default": "standard",
            "flex": "flex",
            "priority": "priority",
        }

        if service_tier not in _map:
            logger.warning("Unknown service_tier value, ignoring", service_tier=repr(service_tier))
            return None

        return _map[service_tier]

    def _build_generation_config(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
        *,
        system_instruction: str | None,
        tools_input: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> genai_types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` from the OpenAI request parameters.

        The optional *tool_choice* argument overrides ``params.tool_choice`` when
        the caller has already resolved a value (e.g. from the deprecated
        ``function_call`` parameter).
        """
        config_kwargs = self._collect_sampling_params(params)

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools_input:
            config_kwargs["tools"] = [genai_types.Tool(**tool) for tool in tools_input]

        resolved_tool_choice = tool_choice if tool_choice is not None else params.tool_choice
        tool_config = self._build_tool_config(resolved_tool_choice)
        if tool_config is not None:
            config_kwargs["tool_config"] = tool_config

        thinking_config = self._build_thinking_config(params.reasoning_effort)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config

        gemini_service_tier = self._convert_service_tier(params.service_tier)
        if gemini_service_tier is not None:
            config_kwargs["service_tier"] = gemini_service_tier

        if params.model_extra:
            config_kwargs.update(params.model_extra)

        return genai_types.GenerateContentConfig(**config_kwargs)

    async def _stream_chat_completion(
        self,
        client: Client,
        provider_model_id: str,
        contents: Any,
        config: genai_types.GenerateContentConfig,
        model: str,
        stream_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        stream = await client.aio.models.generate_content_stream(
            model=provider_model_id,
            contents=contents,
            config=config,
        )
        completion_id = converters.generate_completion_id()
        include_usage = (stream_options or {}).get("include_usage") is True

        async def _iter() -> AsyncIterator[OpenAIChatCompletionChunk]:
            is_first_chunk = True
            last_chunk: Any = None
            async for chunk in stream:
                yield converters.convert_gemini_stream_chunk_to_openai(
                    chunk=chunk,
                    model=model,
                    completion_id=completion_id,
                    is_first_chunk=is_first_chunk,
                )
                is_first_chunk = False
                last_chunk = chunk

            # When include_usage=True, emit a final usage-only chunk per OpenAI spec.
            # This chunk has empty choices and usage populated.
            if include_usage and last_chunk is not None:
                usage = converters.extract_usage(last_chunk)
                yield OpenAIChatCompletionChunk(
                    id=completion_id,
                    choices=[],
                    created=int(time.time()),
                    model=model,
                    usage=usage,
                )

        return _iter()

    async def _stream_completion(
        self,
        client: Client,
        provider_model_id: str,
        contents: Any,
        config: genai_types.GenerateContentConfig,
        model: str,
        stream_options: dict[str, Any] | None = None,
        *,
        completion_id: str | None = None,
        choice_index_offset: int = 0,
    ) -> AsyncIterator[OpenAICompletion]:
        """Stream text completions via Gemini's generate_content_stream."""
        # NOTE: Unlike _stream_chat_completion, we cannot emit a usage-only final
        # chunk because OpenAICompletion requires min_length=1 choices and has no
        # usage field. The stream_options parameter is still accepted and wired
        # through get_stream_options_for_telemetry() for OpenTelemetry span
        # consistency. See OpenAICompletion in ogx_api/inference/models.py.
        stream = await client.aio.models.generate_content_stream(
            model=provider_model_id,
            contents=contents,
            config=config,
        )
        resolved_completion_id = completion_id if completion_id is not None else converters.generate_completion_id()

        async def _iter() -> AsyncIterator[OpenAICompletion]:
            async for chunk in stream:
                yield converters.convert_gemini_stream_chunk_to_openai_completion(
                    chunk=chunk,
                    model=model,
                    completion_id=resolved_completion_id,
                    index_offset=choice_index_offset,
                )

        return _iter()

    def _is_model_allowed(self, provider_model_id: str) -> bool:
        if self.config.allowed_models is None:
            return True
        return provider_model_id in self.config.allowed_models

    def _validate_model_allowed(self, provider_model_id: str) -> None:
        if not self._is_model_allowed(provider_model_id):
            raise ValueError(
                f"Model '{provider_model_id}' is not in the allowed models list. "
                f"Allowed models: {self.config.allowed_models}"
            )

    @staticmethod
    async def _localize_image_url(message: OpenAIMessageParam) -> OpenAIMessageParam:
        """Download HTTP image URLs and convert to data URIs for Gemini compatibility."""
        if isinstance(message.content, list):
            for content_part in message.content:
                if (
                    content_part.type == "image_url"
                    and content_part.image_url
                    and content_part.image_url.url
                    and "http" in content_part.image_url.url
                ):
                    localize_result = await localize_image_content(content_part.image_url.url)
                    if localize_result is None:
                        raise ValueError(f"Failed to localize image content from URL: {content_part.image_url.url}")
                    content, fmt = localize_result
                    content_part.image_url.url = f"data:image/{fmt};base64,{base64.b64encode(content).decode('utf-8')}"

        return message

    @staticmethod
    def _warn_unsupported_chat_params(params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        """Log warnings/debug messages for parameters with no Gemini equivalent."""
        if params.logit_bias is not None:
            logger.warning("VertexAI does not support logit_bias; this parameter will be ignored.")
        if params.parallel_tool_calls is False:
            logger.warning("VertexAI does not support disabling parallel tool calls; this parameter will be ignored.")
        # service_tier is handled by _build_generation_config; no warning needed.
        if params.prompt_cache_key is not None:
            logger.warning("VertexAI does not support prompt_cache_key; this parameter will be ignored.")
        if params.user is not None:
            logger.debug("VertexAI chat completion ignores the 'user' parameter (it is used in embeddings requests).")
        if params.safety_identifier is not None:
            logger.debug("VertexAI does not support safety_identifier; this parameter will be ignored.")

    def _resolve_deprecated_tools(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> tuple[Any, Any]:
        """Resolve deprecated ``functions``/``function_call`` to ``tools``/``tool_choice``."""
        tools = params.tools
        tool_choice = params.tool_choice

        if params.functions and not tools:
            logger.warning("'functions' parameter is deprecated; convert to 'tools' format instead.")
            tools = converters.convert_deprecated_functions_to_tools(params.functions)
        if params.function_call is not None and tool_choice is None:
            logger.warning("'function_call' parameter is deprecated; use 'tool_choice' instead.")
            tool_choice = converters.convert_deprecated_function_call_to_tool_choice(params.function_call)

        return tools, tool_choice

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)
        client = self._get_client()

        self._warn_unsupported_chat_params(params)
        tools, tool_choice = self._resolve_deprecated_tools(params)

        messages = [await self._localize_image_url(message) for message in params.messages]
        system_instruction, contents = converters.convert_openai_messages_to_gemini(messages)
        tools_input = converters.convert_openai_tools_to_gemini(tools)
        config = self._build_generation_config(
            params,
            system_instruction=system_instruction,
            tools_input=tools_input,
            tool_choice=tool_choice,
        )

        request_contents = cast(Any, contents)

        stream_options = get_stream_options_for_telemetry(params.stream_options, params.stream or False)

        if params.stream:
            return await self._stream_chat_completion(
                client,
                provider_model_id,
                request_contents,
                config,
                params.model,
                stream_options=stream_options,
            )

        response = await client.aio.models.generate_content(
            model=provider_model_id,
            contents=request_contents,
            config=config,
        )
        return converters.convert_gemini_response_to_openai(response=response, model=params.model)

    @staticmethod
    def _validate_completion_prompt(prompt: Any) -> list[str]:
        """Validate and normalize a completion prompt to a list of strings."""
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            return [str(p) for p in prompt]
        raise ValueError(
            "VertexAI text completions only support string or list-of-string prompts. "
            "Token array prompts (list[int] or list[list[int]]) are not supported."
        )

    @staticmethod
    def _warn_unsupported_completion_params(params: OpenAICompletionRequestWithExtraBody) -> None:
        """Log warnings/debug messages for unsupported text completion parameters."""
        if params.best_of is not None:
            logger.warning("VertexAI does not support best_of; this parameter will be ignored.")
        if params.suffix is not None:
            logger.warning("VertexAI does not support suffix; this parameter will be ignored.")
        if params.logit_bias is not None:
            logger.warning("VertexAI does not support logit_bias; this parameter will be ignored.")
        if params.user is not None:
            logger.debug("VertexAI text completion ignores the 'user' parameter (it is used in embeddings requests).")

    def _build_completion_config(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> genai_types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` for text completions.

        Uses ``GeminiCompletionSamplingParams`` to map OpenAI field names to
        their Gemini equivalents, mirroring ``_collect_sampling_params`` for
        the chat completion path.
        """
        kwargs = GeminiCompletionSamplingParams.from_openai_params(params).model_dump(exclude_none=True)
        if params.model_extra:
            kwargs.update(params.model_extra)
        return genai_types.GenerateContentConfig(**kwargs)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        prompts = self._validate_completion_prompt(params.prompt)
        self._warn_unsupported_completion_params(params)

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)
        client = self._get_client()
        config = self._build_completion_config(params)

        if params.stream:
            stream_options = get_stream_options_for_telemetry(params.stream_options, params.stream or False)
            shared_completion_id = converters.generate_completion_id()

            async def _multi_prompt_stream() -> AsyncIterator[OpenAICompletion]:
                for i, prompt in enumerate(prompts):
                    if params.echo:
                        yield OpenAICompletion(
                            id=shared_completion_id,
                            choices=[
                                converters.OpenAICompletionChoice(
                                    text=prompt,
                                    finish_reason="stop",
                                    index=i,
                                )
                            ],
                            model=params.model,
                            created=int(time.time()),
                        )
                    contents = converters.convert_completion_prompt_to_contents(prompt)
                    per_prompt_stream = await self._stream_completion(
                        client,
                        provider_model_id,
                        contents,
                        config,
                        params.model,
                        stream_options=stream_options,
                    )
                    async for chunk in per_prompt_stream:
                        n = params.n or 1
                        yield chunk.model_copy(
                            update={
                                "id": shared_completion_id,
                                "choices": [
                                    choice.model_copy(update={"index": i * n + j})
                                    for j, choice in enumerate(chunk.choices)
                                ],
                            }
                        )

            return _multi_prompt_stream()

        all_choices: list[Any] = []
        for prompt in prompts:
            contents = converters.convert_completion_prompt_to_contents(prompt)
            request_contents = cast(Any, contents)
            response = await client.aio.models.generate_content(
                model=provider_model_id,
                contents=request_contents,
                config=config,
            )
            result = converters.convert_gemini_response_to_openai_completion(
                response, model=params.model, prompt=prompt
            )
            for choice in result.choices:
                text = (prompt + choice.text) if params.echo else choice.text
                all_choices.append(
                    converters.OpenAICompletionChoice(
                        text=text,
                        finish_reason=choice.finish_reason,
                        index=len(all_choices),
                    )
                )

        return OpenAICompletion(
            id=converters.generate_completion_id(),
            choices=all_choices,
            created=int(time.time()),
            model=params.model,
        )

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        validate_embeddings_input_is_text(params)

        # Unlike chat completions (which merge model_extra into GenerateContentConfig),
        # EmbedContentConfig has a limited API surface that does not support arbitrary
        # passthrough. Log and ignore extra body parameters rather than silently dropping.
        if params.model_extra:
            logger.debug(
                "VertexAI embeddings does not support extra body parameters; model_extra will be ignored: %s",
                list(params.model_extra.keys()),
            )

        provider_model_id = await self._get_provider_model_id(params.model)
        self._validate_model_allowed(provider_model_id)
        client = self._get_client()

        config_kwargs: dict[str, Any] = {}
        if params.dimensions is not None:
            config_kwargs["output_dimensionality"] = params.dimensions
        if params.user is not None:
            config_kwargs["labels"] = {"user": params.user}
        config = genai_types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

        response = await client.aio.models.embed_content(
            model=provider_model_id,
            contents=cast(Any, params.input),
            config=config,
        )

        data = []
        embeddings = cast(list[Any], response.embeddings or [])
        for i, embedding in enumerate(embeddings):
            if params.encoding_format == "base64":
                values = embedding.value
                float_bytes = struct.pack(f"{len(values)}f", *values)
                embedding_value: list[float] | str = base64.b64encode(float_bytes).decode("ascii")
            else:
                embedding_value = embedding.value
            data.append(OpenAIEmbeddingData(embedding=embedding_value, index=i))

        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        total_tokens = getattr(usage_meta, "total_token_count", 0) or 0
        usage = OpenAIEmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
        return OpenAIEmbeddingsResponse(data=data, model=params.model, usage=usage)

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        _ = request
        raise NotImplementedError("VertexAI rerank not yet implemented")
