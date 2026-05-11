# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Connectors API implementation."""

import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.core.connectors.connectors import (
    ConnectorServiceConfig,
    ConnectorServiceImpl,
)
from ogx.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from ogx.core.storage.kvstore import register_kvstore_backends
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx_api import (
    Connector,
    ConnectorNotFoundError,
    ConnectorType,
    GetConnectorRequest,
    OpenAIResponseInputToolMCP,
    ToolDef,
)

# --- Fixtures ---


@pytest.fixture
async def connector_service(tmp_path_factory):
    """Create a ConnectorServiceImpl with a real SQL store."""
    unique_id = f"connector_store_{random.randint(1, 1000000)}"
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"{unique_id}.db")
    sql_db_path = str(temp_dir / f"{unique_id}_sql.db")

    from ogx.core.datatypes import StackConfig

    storage = StorageConfig(
        backends={
            "kv_test": SqliteKVStoreConfig(db_path=db_path),
            "sql_test": SqliteSqlStoreConfig(db_path=sql_db_path),
        },
        stores=ServerStoresConfig(
            metadata=KVStoreReference(backend="kv_test", namespace="registry"),
            inference=InferenceStoreReference(backend="sql_test", table_name="inference"),
            conversations=SqlStoreReference(backend="sql_test", table_name="conversations"),
            prompts=SqlStoreReference(backend="sql_test", table_name="prompts"),
            connectors=SqlStoreReference(backend="sql_test", table_name="connectors"),
        ),
    )

    register_kvstore_backends({"kv_test": storage.backends["kv_test"]})
    register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})

    mock_run_config = StackConfig(
        distro_name="test-distribution",
        apis=[],
        providers={},
        storage=storage,
    )
    config = ConnectorServiceConfig(config=mock_run_config)
    service = ConnectorServiceImpl(config)
    await service.initialize()

    yield service


@pytest.fixture
def sample_tool_def():
    """Create a sample ToolDef for testing."""
    return ToolDef(
        name="get_weather",
        description="Get weather for a location",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
        output_schema={"type": "object"},
    )


@pytest.fixture
def mock_connectors_api():
    """Create a mock connectors API."""
    api = AsyncMock()
    return api


@pytest.fixture
def sample_connector():
    """Create a sample connector."""
    return Connector(
        connector_id="my-mcp-server",
        connector_type=ConnectorType.MCP,
        url="http://localhost:8080/mcp",
        server_label="My MCP Server",
        server_name="Test Server",
    )


# --- register_connector tests ---


class TestRegisterConnector:
    """Tests for register_connector method."""

    async def test_register_new_connector(self, connector_service):
        """Test registering a new connector creates it in the store."""
        result = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            server_label="My MCP",
        )

        assert result.connector_id == "my-mcp"
        assert result.connector_type == ConnectorType.MCP
        assert result.url == "http://localhost:8080/mcp"
        assert result.server_label == "My MCP"

    async def test_register_connector_different_config_updates(self, connector_service):
        """Attempting to update an existing connector via config should update the existing connector regardless of the source."""
        _ = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            server_label="Original Label",
        )

        _ = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://different-host:9090/mcp",
            server_label="Original Label",
        )

        row = await connector_service.sql_store.fetch_one(
            table="connectors",
            where={"id": "my-mcp"},
        )
        assert row is not None
        connector = Connector.model_validate(row["connector_data"])
        assert connector.url == "http://different-host:9090/mcp"


# --- get_connector tests ---


class TestGetConnector:
    """Tests for get_connector method."""

    async def test_get_connector_not_found(self, connector_service):
        """Test getting a non-existent connector raises error."""
        with pytest.raises(ConnectorNotFoundError) as exc_info:
            await connector_service.get_connector(GetConnectorRequest(connector_id="non-existent"))

        assert "non-existent" in str(exc_info.value)

    async def test_get_connector_returns_with_server_info(self, connector_service):
        """Test getting a connector fetches MCP server info."""
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
        )

        mock_server_info = MagicMock()
        mock_server_info.name = "Test MCP Server"
        mock_server_info.description = "A test server"
        mock_server_info.version = "1.0.0"

        with patch("ogx.core.connectors.connectors.get_mcp_server_info") as mock_get_info:
            mock_get_info.return_value = mock_server_info

            result = await connector_service.get_connector(GetConnectorRequest(connector_id="my-mcp"))

        assert result.connector_id == "my-mcp"
        assert result.server_name == "Test MCP Server"
        assert result.server_description == "A test server"
        assert result.server_version == "1.0.0"

    async def test_get_connector_with_authorization(self, connector_service):
        """Test that authorization is passed to MCP server."""
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
        )

        mock_server_info = MagicMock()
        mock_server_info.name = "Server"
        mock_server_info.description = None
        mock_server_info.version = None

        with patch("ogx.core.connectors.connectors.get_mcp_server_info") as mock_get_info:
            mock_get_info.return_value = mock_server_info

            await connector_service.get_connector(
                GetConnectorRequest(connector_id="my-mcp"), authorization="Bearer token123"
            )

            mock_get_info.assert_called_once_with(
                "http://localhost:8080/mcp",
                authorization="Bearer token123",
            )


# --- list_connectors tests ---


class TestListConnectors:
    """Tests for list_connectors method."""

    async def test_list_connectors_empty(self, connector_service):
        """Test listing connectors when none exist."""
        result = await connector_service.list_connectors()

        assert result.data == []

    async def test_list_connectors_returns_all(self, connector_service):
        """Test listing returns all registered connectors."""
        await connector_service.register_connector(
            connector_id="mcp-1",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8081/mcp",
            server_label="MCP Server 1",
        )
        await connector_service.register_connector(
            connector_id="mcp-2",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8082/mcp",
            server_label="MCP Server 2",
        )
        await connector_service.register_connector(
            connector_id="mcp-3",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8083/mcp",
        )

        result = await connector_service.list_connectors()

        assert len(result.data) == 3
        connector_ids = {c.connector_id for c in result.data}
        assert connector_ids == {"mcp-1", "mcp-2", "mcp-3"}

    async def test_list_connectors_after_unregister(self, connector_service):
        """Test that unregistered connectors are not listed."""
        await connector_service.register_connector(
            connector_id="keep-me",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8081/mcp",
        )
        await connector_service.register_connector(
            connector_id="remove-me",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8082/mcp",
        )

        await connector_service.unregister_connector("remove-me")

        result = await connector_service.list_connectors()

        assert len(result.data) == 1
        assert result.data[0].connector_id == "keep-me"


# --- unregister_connector tests ---


class TestUnregisterConnector:
    """Tests for unregister_connector method."""

    async def test_unregister_existing_connector(self, connector_service):
        """Test unregistering an existing connector removes it from store."""
        await connector_service.register_connector(
            connector_id="to-remove",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
        )

        row = await connector_service.sql_store.fetch_one(
            table="connectors",
            where={"id": "to-remove"},
        )
        assert row is not None

        await connector_service.unregister_connector("to-remove")

        row = await connector_service.sql_store.fetch_one(
            table="connectors",
            where={"id": "to-remove"},
        )
        assert row is None

    async def test_unregister_nonexistent_connector_does_not_raise(self, connector_service):
        """Test unregistering a non-existent connector doesn't raise an error."""
        await connector_service.unregister_connector("does-not-exist")


# --- OpenAIResponseInputToolMCP validation tests ---


class TestMCPToolValidation:
    """Tests for MCP tool input validation."""

    def test_mcp_tool_requires_server_url_or_connector_id(self):
        """Test that either server_url or connector_id must be provided."""
        with pytest.raises(ValueError, match="server_url.*connector_id"):
            OpenAIResponseInputToolMCP(
                type="mcp",
                server_label="test",
            )

    def test_mcp_tool_accepts_server_url_only(self):
        """Test that server_url alone is valid."""
        tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            server_url="http://localhost:8080/mcp",
        )
        assert tool.server_url == "http://localhost:8080/mcp"
        assert tool.connector_id is None

    def test_mcp_tool_accepts_connector_id_only(self):
        """Test that connector_id alone is valid."""
        tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            connector_id="my-connector",
        )
        assert tool.connector_id == "my-connector"
        assert tool.server_url is None

    def test_mcp_tool_accepts_both_server_url_and_connector_id(self):
        """Test that both can be provided (server_url takes precedence)."""
        tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            server_url="http://localhost:8080/mcp",
            connector_id="my-connector",
        )
        assert tool.server_url == "http://localhost:8080/mcp"
        assert tool.connector_id == "my-connector"


# --- connector_id resolution tests ---


class TestConnectorIdResolution:
    """Tests for the resolve_mcp_connector_id helper function."""

    async def test_connector_id_resolved_to_server_url(self, mock_connectors_api, sample_connector):
        """Test that connector_id is resolved to server_url via connectors API."""
        from ogx.providers.inline.responses.builtin.responses.streaming import (
            resolve_mcp_connector_id,
        )

        mock_connectors_api.get_connector.return_value = sample_connector

        mcp_tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            connector_id="my-mcp-server",
        )

        resolved_tool = await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)

        assert resolved_tool.server_url == "http://localhost:8080/mcp"
        mock_connectors_api.get_connector.assert_called_once_with(GetConnectorRequest(connector_id="my-mcp-server"))

    async def test_server_url_not_overwritten_when_provided(self, mock_connectors_api):
        """Test that existing server_url is not overwritten even if connector_id provided."""
        from ogx.providers.inline.responses.builtin.responses.streaming import (
            resolve_mcp_connector_id,
        )

        mcp_tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            server_url="http://original-server:8080/mcp",
            connector_id="my-mcp-server",
        )

        resolved_tool = await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)

        assert resolved_tool.server_url == "http://original-server:8080/mcp"
        mock_connectors_api.get_connector.assert_not_called()

    async def test_connector_id_resolution_propagates_not_found_error(self, mock_connectors_api):
        """Test that ConnectorNotFoundError propagates when connector doesn't exist."""
        from ogx.providers.inline.responses.builtin.responses.streaming import (
            resolve_mcp_connector_id,
        )

        mock_connectors_api.get_connector.side_effect = ConnectorNotFoundError("unknown-connector")

        mcp_tool = OpenAIResponseInputToolMCP(
            type="mcp",
            server_label="test",
            connector_id="unknown-connector",
        )

        with pytest.raises(ConnectorNotFoundError):
            await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)
