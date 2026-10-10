# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.access_control.access_control import is_action_allowed
from ogx.core.datatypes import ModelWithOwner
from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx_api import (
    ModelNotFoundError,
    ModelType,
    ModelTypeError,
    RoutingTable,
    SystemOne,
    SystemOneRequest,
    SystemOneResponse,
)

logger = get_logger(name=__name__, category="core::routers")


class SystemOneRouter(SystemOne):
    """Routes a /v1/systemone judgment request to a provider based on the requested model."""

    def __init__(self, routing_table: RoutingTable) -> None:
        logger.debug("Initializing SystemOneRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("SystemOneRouter.initialize")

    async def shutdown(self) -> None:
        logger.debug("SystemOneRouter.shutdown")

    async def _get_model_provider(self, model_id: str) -> tuple[SystemOne, str]:
        model = await self.routing_table.get_object_by_identifier("model", model_id)
        if model:
            if model.model_type != ModelType.systemone:
                raise ModelTypeError(model_id, model.model_type, ModelType.systemone)
            provider = await self.routing_table.get_provider_impl(model.identifier)
            return provider, model.provider_resource_id

        # Handle the provider_id/provider_resource_id fallback form, same as InferenceRouter.
        splits = model_id.split("/", maxsplit=1)
        if len(splits) != 2:
            raise ModelNotFoundError(model_id)
        provider_id, provider_resource_id = splits

        if provider_id not in self.routing_table.impls_by_provider_id:
            raise ModelNotFoundError(model_id)

        temp_model = ModelWithOwner(
            identifier=model_id,
            provider_id=provider_id,
            provider_resource_id=provider_resource_id,
            model_type=ModelType.systemone,
            metadata={},
        )
        user = get_authenticated_user()
        if not is_action_allowed(self.routing_table.policy, "read", temp_model, user):
            raise ModelNotFoundError(model_id)

        return self.routing_table.impls_by_provider_id[provider_id], provider_resource_id

    async def judge(self, request: SystemOneRequest) -> SystemOneResponse:
        provider, provider_resource_id = await self._get_model_provider(request.model)
        request = request.model_copy(update={"model": provider_resource_id})
        return await provider.judge(request)
