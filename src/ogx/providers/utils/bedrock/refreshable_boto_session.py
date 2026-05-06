# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import datetime
from time import time
from uuid import uuid4

from boto3 import Session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session

from ogx.providers.utils.bedrock.config import DEFAULT_SESSION_TTL


class RefreshableBotoSession:
    """
    Wraps a boto3 session so credentials refresh automatically before they expire.

    Use this when you need a long-lived boto3 client (e.g. a cached bedrock-runtime
    client) without worrying about STS credentials timing out mid-request.
    """

    def __init__(
        self,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        sts_arn: str | None = None,
        web_identity_token_file: str | None = None,
        session_name: str | None = None,
        session_ttl: int = DEFAULT_SESSION_TTL,
    ):
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.profile_name = profile_name
        self.sts_arn = sts_arn
        self.web_identity_token_file = web_identity_token_file
        self.session_name = session_name or uuid4().hex
        self.session_ttl = session_ttl

    def __get_session_credentials(self):
        session_args = {
            "region_name": self.region_name,
            "profile_name": self.profile_name,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
        }
        session_args = {k: v for k, v in session_args.items() if v is not None}
        session = Session(**session_args)

        if self.sts_arn:
            sts_client = session.client(service_name="sts", region_name=self.region_name)

            if self.web_identity_token_file:
                with open(self.web_identity_token_file) as f:
                    web_identity_token = f.read().strip()

                response = sts_client.assume_role_with_web_identity(
                    RoleArn=self.sts_arn,
                    RoleSessionName=self.session_name,
                    WebIdentityToken=web_identity_token,
                    DurationSeconds=self.session_ttl,
                ).get("Credentials")
            else:
                response = sts_client.assume_role(
                    RoleArn=self.sts_arn,
                    RoleSessionName=self.session_name,
                    DurationSeconds=self.session_ttl,
                ).get("Credentials")

            credentials = {
                "access_key": response.get("AccessKeyId"),
                "secret_key": response.get("SecretAccessKey"),
                "token": response.get("SessionToken"),
                "expiry_time": response.get("Expiration").isoformat(),
            }
        else:
            session_credentials = session.get_credentials().get_frozen_credentials()
            credentials = {
                "access_key": session_credentials.access_key,
                "secret_key": session_credentials.secret_key,
                "token": session_credentials.token,
                "expiry_time": datetime.datetime.fromtimestamp(time() + self.session_ttl, datetime.UTC).isoformat(),
            }

        return credentials

    def refreshable_session(self) -> Session:
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=self.__get_session_credentials(),
            refresh_using=self.__get_session_credentials,
            method="sts-assume-role",
        )

        session = get_session()
        session._credentials = refreshable_credentials
        session.set_config_variable("region", self.region_name)
        autorefresh_session = Session(botocore_session=session)

        return autorefresh_session
