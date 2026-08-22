# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.access_control.datatypes import AccessRule
from ogx.core.datatypes import StackConfig
from ogx.core.storage.sqlstore.authorized_sqlstore import authorized_sqlstore
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
    ServiceNotEnabledError,
    ToolDef,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="connectors")

TABLE_CONNECTORS = "connectors"


class ConnectorServiceConfig(BaseModel):
    """Configuration for the built-in connector service."""

    config: StackConfig = Field(..., description="Stack run configuration for resolving persistence")
    policy: list[AccessRule] = []


async def get_provider_impl(config: ConnectorServiceConfig) -> ConnectorServiceImpl:
    """Get the connector service implementation."""
    impl = ConnectorServiceImpl(config)
    return impl


class ConnectorServiceImpl(Connectors):
    """Built-in connector service implementation using AuthorizedSqlStore."""

    def __init__(self, config: ConnectorServiceConfig):
        self.config = config
        self.policy = config.policy

        connectors_ref = config.config.storage.stores.connectors
        if not connectors_ref:
            raise ServiceNotEnabledError("storage.stores.connectors")

        self._connectors_ref = connectors_ref

    async def initialize(self) -> None:
        """Initialize the connector service."""
        self.sql_store = await authorized_sqlstore(self._connectors_ref, self.policy)
        await self.sql_store.create_table(
            TABLE_CONNECTORS,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "connector_type": ColumnType.STRING,
                "url": ColumnType.STRING,
                "server_label": ColumnType.STRING,
                "server_name": ColumnType.STRING,
                "server_description": ColumnType.STRING,
                "connector_data": ColumnType.JSON,
            },
        )

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

        existing_row = await self.sql_store.fetch_one(
            table=TABLE_CONNECTORS,
            where={"id": connector_id},
        )

        if existing_row:
            existing_connector = self._row_to_connector(existing_row)
            if connector == existing_connector:
                logger.info(
                    "Connector already exists; skipping registration",
                    connector_id=connector_id,
                )
                return existing_connector

        connector_data = connector.model_dump()
        row_data = {
            "id": connector_id,
            "connector_type": connector_type.value,
            "url": url,
            "server_label": server_label,
            "server_name": server_name,
            "server_description": server_description,
            "connector_data": connector_data,
        }

        if existing_row:
            await self.sql_store.update(
                table=TABLE_CONNECTORS,
                data=row_data,
                where={"id": connector_id},
            )
        else:
            await self.sql_store.insert(
                table=TABLE_CONNECTORS,
                data=row_data,
            )

        return connector

    async def unregister_connector(self, connector_id: str) -> None:
        """Unregister a connector."""
        existing_row = await self.sql_store.fetch_one(
            table=TABLE_CONNECTORS,
            where={"id": connector_id},
        )
        if not existing_row:
            return
        await self.sql_store.delete(
            table=TABLE_CONNECTORS,
            where={"id": connector_id},
        )

    async def get_connector(
        self,
        request: GetConnectorRequest,
        authorization: str | None = None,
    ) -> Connector:
        """Get a connector by its ID."""

        row = await self.sql_store.fetch_one(
            table=TABLE_CONNECTORS,
            where={"id": request.connector_id},
        )
        if not row:
            raise ConnectorNotFoundError(request.connector_id)

        connector = self._row_to_connector(row)

        server_info = await get_mcp_server_info(connector.url, authorization=authorization)
        connector.server_name = server_info.name
        connector.server_description = server_info.description
        connector.server_version = server_info.version
        return connector

    async def list_connectors(self) -> ListConnectorsResponse:
        """List all connectors."""
        results = await self.sql_store.fetch_all(table=TABLE_CONNECTORS)
        connectors = [self._row_to_connector(row) for row in results.data]
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

    def _row_to_connector(self, row: dict[str, Any]) -> Connector:
        connector_data = row.get("connector_data", {})
        if connector_data:
            return Connector.model_validate(connector_data)
        return Connector(
            connector_id=row["id"],
            connector_type=ConnectorType(row["connector_type"]),
            url=row["url"],
            server_label=row.get("server_label"),
            server_name=row.get("server_name"),
            server_description=row.get("server_description"),
        )

    async def shutdown(self) -> None:
        """Shutdown the connector service."""
        pass
