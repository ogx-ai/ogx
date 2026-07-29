# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from ogx.providers.remote.inference.bedrock.config import BedrockConfig

import httpx
from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError
from pydantic import PrivateAttr

from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.responses.types import (
    AssistantMessageWithReasoning,
)
from ogx.providers.utils.inference.http_client import (
    build_network_client_kwargs,
    network_config_fingerprint,
    set_client_network_fingerprint,
)
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    InternalServerError,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionWithReasoning,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

logger = get_logger(name=__name__, category="inference::bedrock")

_DEFAULT_REGION = "us-east-2"
_BEDROCK_RUNTIME_URL = "https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
_BEDROCK_MANTLE_URL = "https://bedrock-mantle.{region}.api.aws/v1"


class BedrockInferenceAdapter(OpenAIMixin):
    """
    Adapter for AWS Bedrock's OpenAI-compatible API endpoints.

    Supports Llama models across regions and GPT-OSS models (us-west-2 only).

    Authentication modes:
    1. AWS credential chain (recommended): Leave aws_bedrock_bearer_token unset
       and configure AWS credentials for the server environment
       - Web Identity Federation (IRSA, GitHub Actions OIDC)
       - IAM roles (EC2, ECS, Lambda)
       - AWS profiles
       - Static credentials
    2. Bearer token (optional compatibility mode): Set aws_bedrock_bearer_token in
       config or pass it per request in x-ogx-provider-data

    When using AWS credential chain, requests are signed using SigV4 with the
    "bedrock" signing name (note: the endpoint hostname uses "bedrock-runtime",
    but SigV4 credential scope uses the signing name "bedrock").

    Web Identity Federation Examples:

    Kubernetes/OpenShift (IRSA):
        Set these environment variables in your pod spec:
        - AWS_ROLE_ARN=arn:aws:iam::123456789012:role/ogx-role
        - AWS_WEB_IDENTITY_TOKEN_FILE=<path-to-serviceaccount-token>
          Common paths:
          - EKS: /var/run/secrets/eks.amazonaws.com/serviceaccount/token
          - Generic K8s: /var/run/secrets/kubernetes.io/serviceaccount/token
        - AWS_DEFAULT_REGION=us-east-2

    GitHub Actions:
        Use aws-actions/configure-aws-credentials with OIDC:

        permissions:
          id-token: write  # Required for OIDC

        steps:
          - uses: aws-actions/configure-aws-credentials@v4
            with:
              role-to-assume: arn:aws:iam::123456789012:role/github-actions-role
              aws-region: us-east-2

    Credentials are automatically refreshed by boto3 when they expire.

    Models are auto-discovered at startup via two paths:
    - SigV4 auth: uses the ListFoundationModels control-plane API
    - Bearer token auth: queries the mantle endpoint's /v1/models
    Pre-registered models in the config are also supported.
    """

    provider_data_api_key_field: str | None = "aws_bedrock_bearer_token"

    # built once in initialize() so get_extra_client_params() can stay sync;
    # reusing one client also avoids opening a new socket per request
    _sigv4_http_client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _bedrock_client: Any = PrivateAttr(default=None)

    @property
    def _bedrock_config(self) -> "BedrockConfig":
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        if not isinstance(self.config, BedrockConfig):
            raise TypeError(f"Expected BedrockConfig, got {type(self.config)}")
        return self.config

    def get_base_url(self) -> str:
        return _BEDROCK_RUNTIME_URL.format(region=self._bedrock_config.region_name or _DEFAULT_REGION)

    def _should_use_sigv4(self) -> bool:
        # checked per-request so a bearer token in provider data can override SigV4 at runtime
        if self._bedrock_config.has_bearer_token():
            return False

        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.aws_bedrock_bearer_token is not None:
            val = provider_data.aws_bedrock_bearer_token.get_secret_value()
            if val and val.strip():
                return False

        return True

    def _build_sigv4_http_client(self) -> httpx.AsyncClient:
        # lazy import so bearer-token installs don't need boto3/botocore
        from ogx.providers.utils.bedrock.sigv4_auth import BedrockSigV4Auth

        cfg = self._bedrock_config
        sigv4_args: dict[str, Any] = {
            "region": cfg.region_name or "us-east-2",
            "service": "bedrock",  # botocore signing name, not the endpoint prefix "bedrock-runtime"
            "aws_access_key_id": cfg.aws_access_key_id.get_secret_value() if cfg.aws_access_key_id else None,
            "aws_secret_access_key": cfg.aws_secret_access_key.get_secret_value()
            if cfg.aws_secret_access_key
            else None,
            "aws_session_token": cfg.aws_session_token.get_secret_value() if cfg.aws_session_token else None,
            "profile_name": cfg.profile_name,
            "aws_role_arn": cfg.aws_role_arn,
            "aws_web_identity_token_file": cfg.aws_web_identity_token_file,
            "aws_role_session_name": cfg.aws_role_session_name,
            "session_ttl": cfg.session_ttl,
        }
        auth = BedrockSigV4Auth(**{k: v for k, v in sigv4_args.items() if v is not None})
        network_config = cfg.network
        network_kwargs = build_network_client_kwargs(network_config)
        client = httpx.AsyncClient(auth=auth, **network_kwargs)
        if network_config is not None:
            set_client_network_fingerprint(client, network_config_fingerprint(network_config))
        return client

    async def initialize(self) -> None:
        await super().initialize()
        # no request context at init time, so only the static config is available;
        # per-request bearer token overrides are handled in get_extra_client_params()
        if not self._bedrock_config.has_bearer_token():
            self._sigv4_http_client = self._build_sigv4_http_client()
            # separate boto3 client for the bedrock control-plane API (ListFoundationModels)
            try:
                from ogx.providers.utils.bedrock.client import create_bedrock_client

                self._bedrock_client = create_bedrock_client(self._bedrock_config, "bedrock")
            except Exception:
                logger.debug("Could not create Bedrock control-plane client, model discovery will be skipped")

    def get_api_key(self) -> str | None:
        if self._should_use_sigv4():
            # openai sdk requires a non-empty api_key; sigv4_auth will overwrite
            # the resulting "Bearer <NOTUSED>" header with the real SigV4 signature
            return "<NOTUSED>"
        return super().get_api_key()

    def get_extra_client_params(self) -> dict[str, Any]:
        # re-check per request so a runtime bearer token in provider data can bypass sigv4
        if self._sigv4_http_client is not None and self._should_use_sigv4():
            return {"http_client": self._sigv4_http_client}
        return {}

    async def list_provider_model_ids(self) -> Iterable[str]:
        if self._should_use_sigv4():
            # SigV4 path: bedrock-runtime doesn't expose /v1/models,
            # use the control-plane ListFoundationModels API instead
            if self._bedrock_client is None:
                return []
            try:
                response = await asyncio.to_thread(
                    self._bedrock_client.list_foundation_models,
                    byInferenceType="ON_DEMAND",
                )
            except Exception:
                logger.warning("Failed to list Bedrock foundation models", exc_info=True)
                return []
            return [
                m["modelId"] for m in response.get("modelSummaries", []) if m.get("modelLifecycleStatus") == "ACTIVE"
            ]
        # bearer token path: bedrock-runtime doesn't expose /v1/models,
        # but the mantle endpoint does — query it directly
        mantle_url = _BEDROCK_MANTLE_URL.format(region=self._bedrock_config.region_name or _DEFAULT_REGION)
        try:
            client = AsyncOpenAI(base_url=mantle_url, api_key=self.get_api_key())
            return [m.id async for m in client.models.list()]
        except Exception:
            logger.warning("Failed to list models from Bedrock mantle endpoint", exc_info=True)
            return []

    async def check_model_availability(self, model: str) -> bool:
        return True

    async def shutdown(self) -> None:
        if self._sigv4_http_client is not None:
            await asyncio.shield(self._sigv4_http_client.aclose())
            self._sigv4_http_client = None
        await super().shutdown()

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Bedrock's OpenAI-compatible API does not support the /v1/embeddings endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/embeddings endpoint. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Bedrock's OpenAI-compatible API does not support the /v1/completions endpoint."""
        raise NotImplementedError(
            "Bedrock's OpenAI-compatible API does not support /v1/completions endpoint. "
            "Only /v1/chat/completions is supported. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html"
        )

    def _prepare_reasoning_params(self, params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        """Adapt CC request params to match what Bedrock expects for reasoning.

        No-op for now. Override if Bedrock needs specific param adjustments.
        """
        pass

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Chat completion with reasoning support for Bedrock.

        Extracts reasoning from Bedrock's response and wraps it in internal
        types so the Responses layer can read reasoning as a typed field.
        """
        if not params.stream:
            raise NotImplementedError("Non-streaming reasoning is not yet supported for Bedrock")

        params = params.model_copy()
        self._prepare_reasoning_params(params)

        # Bedrock's CC endpoint expects 'reasoning' on assistant messages, but
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

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        use_sigv4 = self._should_use_sigv4()

        try:
            logger.debug("Calling Bedrock OpenAI API", model=params.model, stream=params.stream, sigv4=use_sigv4)
            result = await super().openai_chat_completion(params=params)
            logger.debug("Bedrock API returned", result_type=type(result).__name__ if result is not None else "None")

            if result is None:
                logger.error("Bedrock OpenAI client returned None", model=params.model, stream=params.stream)
                raise RuntimeError(
                    f"Bedrock API returned no response for model '{params.model}'. "
                    "This may indicate the model is not supported or a network/API issue occurred."
                )

            return result
        except (AuthenticationError, PermissionDeniedError) as e:
            # PermissionDeniedError (403) covers SigV4 failures like SignatureDoesNotMatch
            # and AccessDenied — same sanitized path as AuthenticationError (401)
            error_msg = str(e)
            self._handle_auth_error(error_msg, e, use_sigv4=use_sigv4)
        except (RuntimeError, OSError) as e:
            # credential resolution failures (missing AWS creds, unreadable web identity
            # token file, STS errors) should surface as sanitized auth errors, not raw
            # exception messages that may leak internal paths or AWS account details
            if use_sigv4:
                logger.error("AWS Bedrock SigV4 credential resolution failed", error_type=type(e).__name__)
                raise InternalServerError(
                    "Authentication failed because the server could not resolve AWS credentials. "
                    "Please verify that the server has valid AWS credentials configured."
                ) from e
            raise
        except Exception as e:
            logger.error(
                "Unexpected error calling Bedrock API", error_type=type(e).__name__, error=str(e), exc_info=True
            )
            raise

    def _handle_auth_error(self, error_msg: str, original_error: Exception, *, use_sigv4: bool) -> NoReturn:
        if use_sigv4:
            logger.error("AWS Bedrock SigV4 authentication failed")
            raise InternalServerError(
                "Authentication failed because the configured cloud credentials could not authorize this request. "
                "Please verify that the credentials available to the server are valid, unexpired, and allowed to access the requested model."
            ) from original_error

        if "expired" in error_msg.lower() or "Bearer Token has expired" in error_msg:
            logger.error("AWS Bedrock authentication token expired")
            raise InternalServerError(
                "Authentication failed because the provided request credential has expired. "
                "Please refresh the credential and try again, or remove it so the server can use its configured cloud credentials."
            ) from original_error
        logger.error("AWS Bedrock authentication failed")
        raise InternalServerError(
            "Authentication failed because the provided request credential was rejected. "
            "Please verify that the credential is valid, unexpired, and authorized for this request."
        ) from original_error
