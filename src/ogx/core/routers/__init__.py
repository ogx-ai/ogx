# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import (
    AccessRule,
    RoutedProtocol,
    StackConfig,
)
from ogx.core.store import DistributionRegistry
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx_api import Api, RoutingTable


async def get_routing_table_impl(
    api: Api,
    impls_by_provider_id: dict[str, RoutedProtocol],
    _deps,
    dist_registry: DistributionRegistry,
    policy: list[AccessRule],
) -> Any:
    from ..routing_tables.models import ModelsRoutingTable
    from ..routing_tables.shields import ShieldsRoutingTable
    from ..routing_tables.toolgroups import ToolGroupsRoutingTable
    from ..routing_tables.vector_stores import VectorStoresRoutingTable

    api_to_tables = {
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
        "tool_groups": ToolGroupsRoutingTable,
        "vector_stores": VectorStoresRoutingTable,
    }

    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_tables[api.value](impls_by_provider_id, dist_registry, policy)

    await impl.initialize()
    return impl


async def get_auto_router_impl(
    api: Api, routing_table: RoutingTable, deps: dict[str, Any], run_config: StackConfig, policy: list[AccessRule]
) -> Any:
    from .inference import InferenceRouter
    from .safety import SafetyRouter
    from .tool_runtime import ToolRuntimeRouter
    from .vector_io import VectorIORouter

    api_to_routers = {
        "vector_io": VectorIORouter,
        "inference": InferenceRouter,
        "safety": SafetyRouter,
        "tool_runtime": ToolRuntimeRouter,
    }
    if api.value not in api_to_routers:
        raise ValueError(f"API {api.value} not found in router map")

    api_to_dep_impl = {}
    # TODO: move pass configs to routers instead
    if api == Api.inference:
        inference_ref = run_config.storage.stores.inference
        if not inference_ref:
            raise ValueError("storage.stores.inference must be configured in run config")

        inference_store = InferenceStore(
            reference=inference_ref,
            policy=policy,
        )
        await inference_store.initialize()
        api_to_dep_impl["store"] = inference_store
    elif api == Api.vector_io:
        api_to_dep_impl["vector_stores_config"] = run_config.vector_stores
        api_to_dep_impl["inference_api"] = deps.get(Api.inference)
    elif api == Api.safety:
        api_to_dep_impl["safety_config"] = run_config.safety

    impl = api_to_routers[api.value](routing_table, **api_to_dep_impl)

    await impl.initialize()
    return impl
