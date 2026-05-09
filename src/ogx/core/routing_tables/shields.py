# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.datatypes import (
    ShieldWithOwner,
)
from ogx.log import get_logger
from ogx_api import (
    GetShieldRequest,
    ListShieldsResponse,
    RegisterShieldRequest,
    ResourceType,
    Shield,
    Shields,
    UnregisterShieldRequest,
)

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core::routing_tables")


class ShieldsRoutingTable(CommonRoutingTableImpl, Shields):
    """Routing table for managing shield registrations and provider lookups."""

    async def list_shields(self) -> ListShieldsResponse:
        return ListShieldsResponse(data=await self.get_all_with_type(ResourceType.shield.value))

    async def get_shield(self, request: GetShieldRequest) -> Shield:
        shield = await self.get_object_by_identifier("shield", request.identifier)
        if shield is None:
            raise ValueError(f"Shield '{request.identifier}' not found")
        return shield

    async def register_shield(self, request: RegisterShieldRequest) -> Shield:
        provider_shield_id = request.provider_shield_id
        if provider_shield_id is None:
            provider_shield_id = request.shield_id
        provider_id = request.provider_id
        if provider_id is None:
            # If provider_id not specified, use the only provider if it supports this shield type
            if len(self.impls_by_provider_id) == 1:
                provider_id = list(self.impls_by_provider_id.keys())[0]
            else:
                raise ValueError(
                    "No provider specified and multiple providers available. Please specify a provider_id."
                )
        params = request.params
        if params is None:
            params = {}
        shield = ShieldWithOwner(
            identifier=request.shield_id,
            provider_resource_id=provider_shield_id,
            provider_id=provider_id,
            params=params,
        )
        await self.register_object(shield)
        return shield

    async def unregister_shield(self, request: UnregisterShieldRequest) -> None:
        existing_shield = await self.get_shield(GetShieldRequest(identifier=request.identifier))
        await self.unregister_object(existing_shield)
