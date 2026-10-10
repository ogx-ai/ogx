# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import SystemOneRequest, SystemOneResponse


@runtime_checkable
class SystemOne(Protocol):
    """SystemOne

    OGX API for judgment/decision models: models that answer calibrated questions about a
    piece of state (a probability, a choice, or a rubric score) instead of generating text.
    Follows the SystemOne wire format (POST /v1/systemone) that TypeSafe, Ollama, vLLM and
    llama.cpp have converged on.
    """

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        """Answer the request's questions about its state."""
        ...
