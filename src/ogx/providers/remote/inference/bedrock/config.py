# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from pydantic import AliasChoices, BaseModel, Field, SecretStr

from ogx.providers.utils.bedrock.config import BedrockBaseConfig


class BedrockProviderDataValidator(BaseModel):
    """Validates provider-specific request data for AWS Bedrock inference."""

    aws_bedrock_bearer_token: SecretStr | None = Field(
        default=None,
        alias="aws_bedrock_bearer_token",
        validation_alias=AliasChoices("aws_bedrock_bearer_token", "aws_bearer_token_bedrock"),
        description=(
            "Optional per-request bearer token for Amazon Bedrock's OpenAI-compatible runtime. "
            "Leave unset to use the server's AWS credential chain instead."
        ),
    )


class BedrockConfig(BedrockBaseConfig):
    """Configuration for the AWS Bedrock inference provider."""

    auth_credential: SecretStr | None = Field(
        default=None,
        alias="aws_bedrock_bearer_token",
        validation_alias=AliasChoices("aws_bedrock_bearer_token", "api_key"),
        description=(
            "Optional bearer token for Amazon Bedrock's OpenAI-compatible runtime. "
            "Leave unset to use the server's AWS credential chain (recommended)."
        ),
    )
    # Override region_name to default to us-east-2 when unset
    region_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        description="AWS Region for the Bedrock Runtime endpoint",
    )

    def has_bearer_token(self) -> bool:
        """Check if a bearer token is configured."""
        if self.auth_credential is None:
            return False
        token = self.auth_credential.get_secret_value()
        return bool(token and token.strip())

    @classmethod
    def sample_run_config(cls, **kwargs):
        return {
            "aws_bedrock_bearer_token": "${env.AWS_BEDROCK_BEARER_TOKEN:=}",
            "region_name": "${env.AWS_DEFAULT_REGION:=us-east-2}",
            "aws_role_arn": "${env.AWS_ROLE_ARN:=}",
            "aws_web_identity_token_file": "${env.AWS_WEB_IDENTITY_TOKEN_FILE:=}",
        }
