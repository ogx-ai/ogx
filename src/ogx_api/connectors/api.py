# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Connectors API protocol definition.

This module contains the Connectors protocol definition.
Pydantic models are defined in ogx_api.connectors.models.
The FastAPI router is defined in ogx_api.connectors.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from ogx_api.tools import ToolDef

from .models import (
    Connector,
    GetConnectorRequest,
    GetConnectorToolRequest,
    ListConnectorsResponse,
    ListConnectorToolsRequest,
    ListToolsResponse,
)


@runtime_checkable
class Connectors(Protocol):
    """Protocol for connector management operations."""

    async def get_connector(
        self,
        request: GetConnectorRequest,
        authorization: str | None = None,
    ) -> Connector: ...

    async def list_connectors(self) -> ListConnectorsResponse: ...

    async def list_connector_tools(
        self,
        request: ListConnectorToolsRequest,
        authorization: str | None = None,
    ) -> ListToolsResponse: ...

    async def get_connector_tool(
        self,
        request: GetConnectorToolRequest,
        authorization: str | None = None,
    ) -> ToolDef: ...
