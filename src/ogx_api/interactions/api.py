# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .models import (
    GoogleCreateInteractionRequest,
    GoogleInteractionResponse,
    GoogleStreamEvent,
)


@runtime_checkable
class Interactions(Protocol):
    """Protocol for the Google Interactions API."""

    async def create_interaction(
        self,
        request: GoogleCreateInteractionRequest,
    ) -> GoogleInteractionResponse | AsyncIterator[GoogleStreamEvent] | AsyncIterator[str]: ...
