# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shields API protocol definition.

This module contains the Shields protocol for managing shield resources.
"""

from typing import Protocol, runtime_checkable

from .models import (
    GetShieldRequest,
    ListShieldsResponse,
    RegisterShieldRequest,
    Shield,
    UnregisterShieldRequest,
)


@runtime_checkable
class Shields(Protocol):
    """Protocol for managing safety shield resources."""

    async def list_shields(self) -> ListShieldsResponse:
        """List all shields."""
        ...

    async def get_shield(self, request: GetShieldRequest) -> Shield:
        """Get a shield by its identifier."""
        ...

    async def register_shield(self, request: RegisterShieldRequest) -> Shield:
        """Register a shield."""
        ...

    async def unregister_shield(self, request: UnregisterShieldRequest) -> None:
        """Unregister a shield."""
        ...
