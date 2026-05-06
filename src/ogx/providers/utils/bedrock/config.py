# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from pydantic import Field, SecretStr

# 1 hour — matches AWS's default role expiration and minimum recommended TTL
DEFAULT_SESSION_TTL = 3600

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig


class BedrockBaseConfig(RemoteInferenceProviderConfig):
    """Base configuration for AWS Bedrock providers with credential and connection settings."""

    auth_credential: None = Field(default=None, exclude=True)
    aws_access_key_id: SecretStr | None = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("AWS_ACCESS_KEY_ID")) else None,
        description="The AWS access key to use. Default use environment variable: AWS_ACCESS_KEY_ID",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("AWS_SECRET_ACCESS_KEY")) else None,
        description="The AWS secret access key to use. Default use environment variable: AWS_SECRET_ACCESS_KEY",
    )
    aws_session_token: SecretStr | None = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("AWS_SESSION_TOKEN")) else None,
        description="The AWS session token to use. Default use environment variable: AWS_SESSION_TOKEN",
    )
    aws_role_arn: str | None = Field(
        default_factory=lambda: os.getenv("AWS_ROLE_ARN"),
        description="The AWS role ARN to assume. Default use environment variable: AWS_ROLE_ARN",
    )
    aws_web_identity_token_file: str | None = Field(
        default_factory=lambda: os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE"),
        description="The path to the web identity token file. Default use environment variable: AWS_WEB_IDENTITY_TOKEN_FILE",
    )
    aws_role_session_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_ROLE_SESSION_NAME"),
        description="The session name to use when assuming a role. Default use environment variable: AWS_ROLE_SESSION_NAME",
    )
    region_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION"),
        description="The default AWS Region to use, for example, us-west-1 or us-west-2."
        "Default use environment variable: AWS_DEFAULT_REGION",
    )
    profile_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_PROFILE"),
        description="The profile name that contains credentials to use.Default use environment variable: AWS_PROFILE",
    )
    total_max_attempts: int | None = Field(
        default_factory=lambda: int(val) if (val := os.getenv("AWS_MAX_ATTEMPTS")) else None,
        description="An integer representing the maximum number of attempts that will be made for a single request, "
        "including the initial attempt. Default use environment variable: AWS_MAX_ATTEMPTS",
    )
    retry_mode: str | None = Field(
        default_factory=lambda: os.getenv("AWS_RETRY_MODE"),
        description="A string representing the type of retries Boto3 will perform."
        "Default use environment variable: AWS_RETRY_MODE",
    )
    connect_timeout: float | None = Field(
        default_factory=lambda: float(os.getenv("AWS_CONNECT_TIMEOUT", "60")),
        description="The time in seconds till a timeout exception is thrown when attempting to make a connection. "
        "The default is 60 seconds.",
    )
    read_timeout: float | None = Field(
        default_factory=lambda: float(os.getenv("AWS_READ_TIMEOUT", "60")),
        description="The time in seconds till a timeout exception is thrown when attempting to read from a connection."
        "The default is 60 seconds.",
    )
    session_ttl: int | None = Field(
        default_factory=lambda: int(os.getenv("AWS_SESSION_TTL", str(DEFAULT_SESSION_TTL))),
        description="The time in seconds till a session expires. The default is 3600 seconds (1 hour).",
    )

    @classmethod
    def sample_run_config(cls, **kwargs):
        return {}
