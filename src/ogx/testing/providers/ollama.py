# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Ollama provider exception handling for test recording/replay.

Handles native Ollama errors that don't go through the OpenAI SDK.
Most Ollama inference errors go through OpenAIMixin and are already
handled as OpenAI errors.
"""

import ollama as ollama_sdk
from ollama import ResponseError

from ogx.testing.providers._config import ProviderConfig


def create_error(status_code: int, body: dict | None, message: str) -> ResponseError:
    """Reconstruct an Ollama ResponseError from recorded data."""
    return ResponseError(error=message, status_code=status_code)


PROVIDER = ProviderConfig(
    name="ollama",
    sdk_module=ollama_sdk,
    create_error=create_error,
)
