# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
SigV4 authentication for AWS Bedrock OpenAI-compatible endpoint.

This module provides httpx.Auth implementation that signs requests using
AWS Signature Version 4, enabling IAM/STS authentication with the Bedrock
OpenAI-compatible API endpoint.

Supported credential sources (via boto3 credential chain):
- Static credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- Web Identity Federation (AWS_ROLE_ARN, AWS_WEB_IDENTITY_TOKEN_FILE)
- IAM roles (IMDS for EC2, ECS task roles, Lambda execution roles)
- AWS profiles (~/.aws/credentials)

Web Identity Federation enables keyless authentication in:
- Kubernetes/OpenShift with IRSA (IAM Roles for Service Accounts)
- GitHub Actions with OIDC (aws-actions/configure-aws-credentials)
- Any OIDC-compatible identity provider

Environment variables for Web Identity:
    AWS_ROLE_ARN: ARN of the IAM role to assume
    AWS_WEB_IDENTITY_TOKEN_FILE: Path to the OIDC token file
        Common paths:
        - EKS: /var/run/secrets/eks.amazonaws.com/serviceaccount/token
        - Generic Kubernetes: /var/run/secrets/kubernetes.io/serviceaccount/token
        - GitHub Actions: Set automatically by aws-actions/configure-aws-credentials
    AWS_DEFAULT_REGION: AWS region for the Bedrock endpoint

Credentials are automatically refreshed by boto3 when they expire.

References:
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html
- https://github.com/ogx-ai/ogx/issues/4730
- https://github.com/opendatahub-io/ogx-distribution/issues/112
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from ogx.log import get_logger
from ogx.providers.utils.bedrock.config import DEFAULT_SESSION_TTL

logger = get_logger(name=__name__, category="providers")


class BedrockSigV4Auth(httpx.Auth):
    """
    httpx.Auth that signs requests with AWS SigV4.

    Only signs headers that httpx won't touch after signing, to avoid
    signature mismatches. Credential refresh is handled automatically
    by boto3 for temporary credentials (STS, IRSA).
    """

    def __init__(
        self,
        region: str,
        service: str = "bedrock",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        aws_role_arn: str | None = None,
        aws_web_identity_token_file: str | None = None,
        aws_role_session_name: str | None = None,
        session_ttl: int | None = DEFAULT_SESSION_TTL,
    ):
        # service must be "bedrock" (the botocore signing name), not "bedrock-runtime"
        # (the endpoint prefix) — using the wrong one causes SignatureDoesNotMatch
        self._region = region
        self._service = service
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._profile_name = profile_name
        self._aws_role_arn = aws_role_arn
        self._aws_web_identity_token_file = aws_web_identity_token_file
        self._aws_role_session_name = aws_role_session_name
        self._session_ttl = session_ttl or DEFAULT_SESSION_TTL
        self._lock = threading.Lock()
        self._session: Any = None  # boto3.Session | None — Any because boto3 is an optional dep

    def _get_credentials(self) -> Any:
        from ogx.providers.utils.bedrock.refreshable_boto_session import (
            RefreshableBotoSession,
        )

        with self._lock:
            if self._session is None:
                if self._aws_role_arn:
                    self._session = RefreshableBotoSession(
                        region_name=self._region,
                        aws_access_key_id=self._aws_access_key_id,
                        aws_secret_access_key=self._aws_secret_access_key,
                        aws_session_token=self._aws_session_token,
                        profile_name=self._profile_name,
                        sts_arn=self._aws_role_arn,
                        web_identity_token_file=self._aws_web_identity_token_file,
                        session_name=self._aws_role_session_name,
                        session_ttl=self._session_ttl,
                    ).refreshable_session()
                else:
                    import boto3

                    self._session = boto3.Session(
                        region_name=self._region,
                        aws_access_key_id=self._aws_access_key_id,
                        aws_secret_access_key=self._aws_secret_access_key,
                        aws_session_token=self._aws_session_token,
                        profile_name=self._profile_name,
                    )

            credentials = self._session.get_credentials()
            if credentials is None:
                raise RuntimeError(
                    "Failed to load AWS credentials. Ensure AWS credentials are "
                    "configured via environment variables (AWS_ACCESS_KEY_ID, "
                    "AWS_SECRET_ACCESS_KEY), IAM role, or AWS profile."
                )
            return credentials.get_frozen_credentials()

    def _sign_request(self, request: httpx.Request) -> None:
        credentials = self._get_credentials()

        # drop the openai sdk's "Bearer <NOTUSED>" placeholder before signing
        if "authorization" in request.headers:
            del request.headers["authorization"]

        # sign only stable headers — anything httpx might rewrite after this point
        # would invalidate the signature, so we leave those out
        host = request.headers.get("host") or str(request.url.netloc)
        headers_to_sign = {"host": host}

        # only include content-type if the request already has one; injecting a
        # default here would cause a mismatch if httpx sends a different value
        if "content-type" in request.headers:
            headers_to_sign["content-type"] = request.headers["content-type"]

        for header_name in ["x-amz-content-sha256", "x-amz-security-token"]:
            if header_name in request.headers:
                headers_to_sign[header_name] = request.headers[header_name]

        try:
            content = request.content
        except httpx.RequestNotRead:
            content = request.read()

        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=content,
            headers=headers_to_sign,
        )

        signer = SigV4Auth(credentials, self._service, self._region)
        signer.add_auth(aws_request)

        # copy Authorization, X-Amz-Date, and X-Amz-Security-Token back onto the live request
        for key, value in aws_request.headers.items():
            request.headers[key] = value

        logger.debug(
            f"SigV4 signed request: method={request.method}, "
            f"path={request.url.path}, service={self._service}, region={self._region}"
        )

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        self._sign_request(request)
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        # offload to a thread because credential resolution can do IMDS calls or file I/O;
        # shield so a rolling-restart cancellation doesn't abort mid-sign and leave the
        # connection in an inconsistent auth state
        await asyncio.shield(asyncio.to_thread(self._sign_request, request))
        yield request
