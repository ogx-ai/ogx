# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import boto3
from botocore.client import BaseClient
from botocore.config import Config

from ogx.providers.utils.bedrock.config import DEFAULT_SESSION_TTL, BedrockBaseConfig
from ogx.providers.utils.bedrock.refreshable_boto_session import (
    RefreshableBotoSession,
)


def create_bedrock_client(config: BedrockBaseConfig, service_name: str = "bedrock-runtime") -> BaseClient:
    """Creates a boto3 client for Bedrock services with the given configuration.

    Args:
        config: The Bedrock configuration containing AWS credentials and settings
        service_name: The AWS service name to create client for (default: "bedrock-runtime")

    Returns:
        A configured boto3 client
    """
    retries_config = {
        k: v
        for k, v in dict(
            total_max_attempts=config.total_max_attempts,
            mode=config.retry_mode,
        ).items()
        if v is not None
    }
    boto3_config_args = {
        k: v
        for k, v in dict(
            region_name=config.region_name,
            retries=retries_config if retries_config else None,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
        ).items()
        if v is not None
    }
    boto3_config = Config(**boto3_config_args) if boto3_config_args else None

    if config.aws_role_arn:
        # role assumption takes priority — source credentials (if any) are passed in
        # so the refreshable session can use them as the base for assume-role calls
        client = RefreshableBotoSession(
            region_name=config.region_name,
            aws_access_key_id=config.aws_access_key_id.get_secret_value() if config.aws_access_key_id else None,
            aws_secret_access_key=config.aws_secret_access_key.get_secret_value()
            if config.aws_secret_access_key
            else None,
            aws_session_token=config.aws_session_token.get_secret_value() if config.aws_session_token else None,
            profile_name=config.profile_name,
            sts_arn=config.aws_role_arn,
            web_identity_token_file=config.aws_web_identity_token_file,
            session_name=config.aws_role_session_name,
            session_ttl=config.session_ttl or DEFAULT_SESSION_TTL,
        ).refreshable_session()
        return client.client(service_name, config=boto3_config) if boto3_config else client.client(service_name)
    elif config.aws_access_key_id and config.aws_secret_access_key:
        session_args = {
            "aws_access_key_id": config.aws_access_key_id.get_secret_value(),
            "aws_secret_access_key": config.aws_secret_access_key.get_secret_value(),
            "aws_session_token": config.aws_session_token.get_secret_value() if config.aws_session_token else None,
            "region_name": config.region_name,
            "profile_name": config.profile_name,
        }

        # Remove None values
        session_args = {k: v for k, v in session_args.items() if v is not None}

        boto3_session = boto3.session.Session(**session_args)
        return boto3_session.client(service_name, config=boto3_config)
    else:
        session = RefreshableBotoSession(
            region_name=config.region_name,
            profile_name=config.profile_name,
            session_ttl=config.session_ttl or DEFAULT_SESSION_TTL,
        ).refreshable_session()
        return session.client(service_name, config=boto3_config) if boto3_config else session.client(service_name)
