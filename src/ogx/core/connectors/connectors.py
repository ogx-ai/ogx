# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ogx.core.datatypes import StackConfig
from ogx.core.storage.kvstore import KVStore, kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.tools.mcp import get_mcp_server_info, list_mcp_tools
from ogx_api import (
    Connector,
    ConnectorNotFoundError,
    Connectors,
    ConnectorToolNotFoundError,
    ConnectorType,
    GetConnectorRequest,
    GetConnectorToolRequest,
    ListConnectorsResponse,
    ListConnectorToolsRequest,
    ListToolsResponse,
    ToolDef,
)

logger = get_logger(name=__name__, category="connectors")


class ConnectorServiceConfig(BaseModel):
    """Configuration for the built-in connector service."""

    config: StackConfig = Field(..., description="Stack run configuration for resolving persistence")


async def get_provider_impl(config: ConnectorServiceConfig) -> ConnectorServiceImpl:
    """Get the connector service implementation."""
    impl = ConnectorServiceImpl(config)
    return impl


KEY_PREFIX = "connectors:v1:"


class ConnectorServiceImpl(Connectors):
    """Built-in connector service implementation."""

    def __init__(self, config: ConnectorServiceConfig):
        self.config = config
        self.kvstore: KVStore

    def _get_key(self, connector_id: str) -> str:
        """Get the KVStore key for a connector."""
        return f"{KEY_PREFIX}{connector_id}"

    async def initialize(self) -> None:
        """Initialize the connector service."""

        # Use connectors store reference from run config
        connectors_ref = self.config.config.storage.stores.connectors
        if not connectors_ref:
            raise ValueError("storage.stores.connectors must be configured in config")
        self.kvstore = await kvstore_impl(connectors_ref)

    async def register_connector(
        self,
        connector_id: str,
        connector_type: ConnectorType,
        url: str,
        server_label: str | None = None,
        server_name: str | None = None,
        server_description: str | None = None,
    ) -> Connector:
        """Register a new connector"""

        connector = Connector(
            connector_id=connector_id,
            connector_type=connector_type,
            url=url,
            server_label=server_label,
            server_name=server_name,
            server_description=server_description,
        )

        key = self._get_key(connector_id)
        existing_connector_json = await self.kvstore.get(key)

        if existing_connector_json:
            existing_connector = Connector.model_validate_json(existing_connector_json)

            if connector == existing_connector:
                logger.info(
                    "Connector %s already exists; skipping registration",
                    connector_id,
                )
                return existing_connector

        await self.kvstore.set(key, json.dumps(connector.model_dump()))

        return connector

    async def unregister_connector(self, connector_id: str) -> None:
        """Unregister a connector."""
        key = self._get_key(connector_id)
        if not await self.kvstore.get(key):
            return
        await self.kvstore.delete(key)

    async def get_connector(
        self,
        request: GetConnectorRequest,
        authorization: str | None = None,
    ) -> Connector:
        """Get a connector by its ID."""

        connector_json = await self.kvstore.get(self._get_key(request.connector_id))
        if not connector_json:
            raise ConnectorNotFoundError(request.connector_id)
        connector = Connector.model_validate_json(connector_json)

        server_info = await get_mcp_server_info(connector.url, authorization=authorization)
        connector.server_name = server_info.name
        connector.server_description = server_info.description
        connector.server_version = server_info.version
        return connector

    async def list_connectors(self) -> ListConnectorsResponse:
        """List all connectors."""
        values = await self.kvstore.values_in_range(start_key=KEY_PREFIX, end_key=KEY_PREFIX + "\uffff")
        connectors = [Connector.model_validate_json(v) for v in values]
        return ListConnectorsResponse(data=connectors)

    async def get_connector_tool(self, request: GetConnectorToolRequest, authorization: str | None = None) -> ToolDef:
        """Get a tool from a connector."""
        connector_tools = await self.list_connector_tools(
            ListConnectorToolsRequest(connector_id=request.connector_id), authorization=authorization
        )
        for tool in connector_tools.data:
            if tool.name == request.tool_name:
                return tool
        raise ConnectorToolNotFoundError(request.connector_id, request.tool_name)

    async def list_connector_tools(
        self, request: ListConnectorToolsRequest, authorization: str | None = None
    ) -> ListToolsResponse:
        """List tools from a connector."""
        connector = await self.get_connector(
            GetConnectorRequest(connector_id=request.connector_id), authorization=authorization
        )
        tools = await list_mcp_tools(endpoint=connector.url, authorization=authorization)
        return ListToolsResponse(data=tools.data)

    async def shutdown(self) -> None:
        """Shutdown the connector service."""
        await self.kvstore.shutdown()
