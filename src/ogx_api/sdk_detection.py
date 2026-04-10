# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""SDK detection utility for multi-SDK response shaping.

Detects which SDK is making a request based on well-known headers,
allowing shared endpoints like /v1/models to return SDK-appropriate
response formats.
"""

from enum import StrEnum

from fastapi import Request


class SdkType(StrEnum):
    """The SDK type detected from request headers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


def detect_sdk(request: Request) -> SdkType:
    """Detect which SDK is making the request based on headers.

    Detection priority:
    1. anthropic-version header → Anthropic SDK
    2. x-goog-api-key header → Google AI SDK
    3. Default → OpenAI SDK
    """
    headers = request.headers
    if headers.get("anthropic-version"):
        return SdkType.ANTHROPIC
    if headers.get("x-goog-api-key"):
        return SdkType.GOOGLE
    return SdkType.OPENAI
