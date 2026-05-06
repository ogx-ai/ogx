# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from typing import Any

from ogx.log import get_logger
from ogx.telemetry.tool_runtime_metrics import (
    create_tool_metric_attributes,
    tool_duration,
    tool_invocations_total,
)
from ogx_api import (
    URL,
    ListToolDefsResponse,
    ListToolsRequest,
    ToolRuntime,
)

from ..routing_tables.toolgroups import ToolGroupsRoutingTable

logger = get_logger(name=__name__, category="core::routers")


class ToolRuntimeRouter(ToolRuntime):
    """Router that delegates tool runtime operations to the appropriate provider via a routing table."""

    def __init__(
        self,
        routing_table: ToolGroupsRoutingTable,
    ) -> None:
        logger.debug("Initializing ToolRuntimeRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        logger.debug("ToolRuntimeRouter.initialize")
        pass

    async def shutdown(self) -> None:
        logger.debug("ToolRuntimeRouter.shutdown")
        pass

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any], authorization: str | None = None) -> Any:
        logger.debug("ToolRuntimeRouter.invoke_tool", tool_name=tool_name)
        start_time = time.perf_counter()
        metric_attrs = None

        try:
            # Get provider and tool metadata for metrics
            provider = await self.routing_table.get_provider_impl(tool_name)

            # Try to get tool group ID from the routing table cache
            tool_group = None
            try:
                if tool_name in self.routing_table.tool_to_toolgroup:
                    tool_group = self.routing_table.tool_to_toolgroup[tool_name]
            except Exception:
                # If we can't get the tool group, continue without it
                pass

            # Create metric attributes
            metric_attrs = create_tool_metric_attributes(
                tool_group=tool_group,
                tool_name=tool_name,
                provider=getattr(provider, "__provider_id__", None),
            )

            # Execute tool invocation
            result = await provider.invoke_tool(
                tool_name=tool_name,
                kwargs=kwargs,
                authorization=authorization,
            )

            # Record success metrics
            duration = time.perf_counter() - start_time
            success_attrs = {**metric_attrs, "status": "success"}
            tool_invocations_total.add(1, success_attrs)
            tool_duration.record(duration, metric_attrs)

            return result

        except asyncio.CancelledError:
            # Record cancellation metrics
            duration = time.perf_counter() - start_time
            if metric_attrs:
                error_attrs = {**metric_attrs, "status": "error"}
            else:
                error_attrs = create_tool_metric_attributes(
                    tool_name=tool_name,
                    status="error",
                )
            tool_invocations_total.add(1, error_attrs)
            tool_duration.record(duration, error_attrs)
            raise
        except Exception:
            # Record error metrics
            duration = time.perf_counter() - start_time
            if metric_attrs:
                error_attrs = {**metric_attrs, "status": "error"}
            else:
                # If we failed before creating metric_attrs, create minimal attrs
                error_attrs = create_tool_metric_attributes(
                    tool_name=tool_name,
                    status="error",
                )
            tool_invocations_total.add(1, error_attrs)
            tool_duration.record(duration, error_attrs)
            raise

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None, authorization: str | None = None
    ) -> ListToolDefsResponse:
        return await self.routing_table.list_tools(
            ListToolsRequest(toolgroup_id=tool_group_id), authorization=authorization
        )
