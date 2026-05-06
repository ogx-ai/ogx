# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from pydantic import BaseModel, Field, SecretStr

from ogx.providers.utils.bedrock.config import BedrockBaseConfig


class BedrockProviderDataValidator(BaseModel):
    """Validates provider-specific request data for AWS Bedrock inference."""

    aws_bearer_token_bedrock: SecretStr | None = Field(
        default=None,
        description="API Key (Bearer token) for Amazon Bedrock",
    )


class BedrockConfig(BedrockBaseConfig):
    """Configuration for the AWS Bedrock inference provider."""

    auth_credential: SecretStr | None = Field(
        default=None,
        description="Authentication credential for the provider",
        alias="api_key",
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
            "api_key": "${env.AWS_BEARER_TOKEN_BEDROCK:=}",
            "region_name": "${env.AWS_DEFAULT_REGION:=us-east-2}",
            "aws_role_arn": "${env.AWS_ROLE_ARN:=}",
            "aws_web_identity_token_file": "${env.AWS_WEB_IDENTITY_TOKEN_FILE:=}",
        }
