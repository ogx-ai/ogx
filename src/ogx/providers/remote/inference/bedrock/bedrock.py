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
from openai import AuthenticationError, PermissionDeniedError
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


class BedrockInferenceAdapter(OpenAIMixin):
    """
    Adapter for AWS Bedrock's OpenAI-compatible API endpoints.

    Supports Llama models across regions and GPT-OSS models (us-west-2 only).

    Authentication modes:
    1. Bearer token (legacy): Set AWS_BEARER_TOKEN_BEDROCK or api_key in config
    2. AWS credential chain (enterprise): Leave api_key unset, configure AWS creds
       - Web Identity Federation (IRSA, GitHub Actions OIDC)
       - IAM roles (EC2, ECS, Lambda)
       - AWS profiles
       - Static credentials

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

    Note: Bedrock's OpenAI-compatible endpoint does not support /v1/models
    for dynamic model discovery. Models must be pre-registered in the config.
    """

    provider_data_api_key_field: str | None = "aws_bearer_token_bedrock"

    # built once in initialize() so get_extra_client_params() can stay sync;
    # reusing one client also avoids opening a new socket per request
    _sigv4_http_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    @property
    def _bedrock_config(self) -> "BedrockConfig":
        from ogx.providers.remote.inference.bedrock.config import BedrockConfig

        if not isinstance(self.config, BedrockConfig):
            raise TypeError(f"Expected BedrockConfig, got {type(self.config)}")
        return self.config

    def get_base_url(self) -> str:
        region = self._bedrock_config.region_name or "us-east-2"
        return f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"

    def _should_use_sigv4(self) -> bool:
        # checked per-request so a bearer token in provider data can override SigV4 at runtime
        if self._bedrock_config.has_bearer_token():
            return False

        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.aws_bearer_token_bedrock is not None:
            val = provider_data.aws_bearer_token_bedrock.get_secret_value()
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
        # bedrock's openai-compatible endpoint doesn't expose /v1/models
        return []

    async def check_model_availability(self, model: str) -> bool:
        # no /v1/models to query — accept whatever is registered in config
        return True

    async def shutdown(self) -> None:
        if self._sigv4_http_client is not None:
            # shield so stack.py's 5s asyncio.wait_for doesn't abort mid-close and leak a socket
            await asyncio.shield(self._sigv4_http_client.aclose())
            self._sigv4_http_client = None

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
