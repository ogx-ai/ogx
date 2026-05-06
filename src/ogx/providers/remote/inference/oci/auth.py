# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Generator, Mapping
from typing import Any, override

import httpx
import oci
import requests
from oci.config import DEFAULT_LOCATION, DEFAULT_PROFILE

OciAuthSigner = type[oci.signer.AbstractBaseSigner]


class HttpxOciAuth(httpx.Auth):
    """
    Custom HTTPX authentication class that implements OCI request signing.

    This class handles the authentication flow for HTTPX requests by signing them
    using the OCI Signer, which adds the necessary authentication headers for
    OCI API calls.

    Attributes:
        signer (oci.signer.Signer): The OCI signer instance used for request signing
    """

    def __init__(self, signer: OciAuthSigner):
        self.signer = signer

    @override
    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        # Read the request content to handle streaming requests properly
        try:
            content = request.content
        except httpx.RequestNotRead:
            # For streaming requests, we need to read the content first
            content = request.read()

        req = requests.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=content,
        )
        prepared_request = req.prepare()

        # Sign the request using the OCI Signer
        self.signer.do_request_sign(prepared_request)  # type: ignore

        # Update the original HTTPX request with the signed headers
        request.headers.update(prepared_request.headers)

        yield request


class OciInstancePrincipalAuth(HttpxOciAuth):
    """OCI authentication using instance principal credentials."""

    def __init__(self, **kwargs: Mapping[str, Any]):
        self.signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner(**kwargs)


class OciUserPrincipalAuth(HttpxOciAuth):
    """OCI authentication using user principal credentials from a config file."""

    def __init__(self, config_file: str = DEFAULT_LOCATION, profile_name: str = DEFAULT_PROFILE):
        config = oci.config.from_file(config_file, profile_name)
        oci.config.validate_config(config)  # type: ignore
        key_content = ""
        with open(config["key_file"]) as f:
            key_content = f.read()

        self.signer = oci.signer.Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=config.get("key_file"),
            pass_phrase="none",  # type: ignore
            private_key_content=key_content,
        )
