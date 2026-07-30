# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAI provider exception handling for test recording/replay."""

import httpx
import openai as openai_sdk
from openai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from ogx.testing.providers._config import ProviderConfig

_ERROR_BY_STATUS: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
}


def create_error(status_code: int, body: dict | None, message: str) -> APIStatusError:
    """Reconstruct an OpenAI API error from recorded data."""
    error_class = _ERROR_BY_STATUS.get(status_code, APIStatusError)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, json=body or {}, request=request)
    return error_class(message=message, response=response, body=body)


PROVIDER = ProviderConfig(
    name="openai",
    sdk_module=openai_sdk,
    create_error=create_error,
)
