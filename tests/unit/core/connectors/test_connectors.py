# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Connectors API implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.core.connectors.connectors import (
    KEY_PREFIX,
    ConnectorServiceImpl,
)
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
def mock_kvstore():
    """Create a mock KVStore with in-memory storage."""
    storage = {}

    class MockKVStore:
        async def set(self, key, value):
            storage[key] = value

        async def get(self, key):
            return storage.get(key)

        async def delete(self, key):
            del storage[key]

        async def keys_in_range(self, start_key, end_key):
            return [k for k in storage.keys() if start_key <= k < end_key]

        async def values_in_range(self, start_key, end_key):
            return [v for k, v in storage.items() if start_key <= k < end_key]

        async def close(self):
            pass

        @property
        def _storage(self):
            return storage

    return MockKVStore()


@pytest.fixture
async def connector_service(mock_kvstore):
    """Create a ConnectorServiceImpl with mocked dependencies."""
    # Create a minimal mock config - we'll inject the kvstore directly
    mock_config = MagicMock(spec=Connector)

    with patch(
        "ogx.core.connectors.connectors.kvstore_impl",
        return_value=mock_kvstore,
    ):
        service = ConnectorServiceImpl(mock_config)
        service.kvstore = mock_kvstore  # Inject directly
        return service


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

    async def test_register_new_connector(self, connector_service, mock_kvstore):
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

        # Verify stored in kvstore
        stored = await mock_kvstore.get(f"{KEY_PREFIX}my-mcp")
        assert stored is not None

    async def test_register_connector_different_config_updates(self, connector_service, mock_kvstore):
        """Attempting to update an existing connector via config should update the existing connector regardless of the source."""
        # Register the original connector
        _ = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
            server_label="Original Label",
        )

        # Attempt to update with a different URL
        _ = await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://different-host:9090/mcp",
            server_label="Original Label",
        )

        # Existing connector should be returned and updated
        stored = await mock_kvstore.get(f"{KEY_PREFIX}my-mcp")
        persisted = Connector.model_validate_json(stored)
        assert persisted.url == "http://different-host:9090/mcp"


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
        # Register a connector
        await connector_service.register_connector(
            connector_id="my-mcp",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
        )

        # Mock the MCP server info response
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
        # Register multiple connectors
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
        # Register connectors
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

        # Unregister one
        await connector_service.unregister_connector("remove-me")

        result = await connector_service.list_connectors()

        assert len(result.data) == 1
        assert result.data[0].connector_id == "keep-me"


# --- unregister_connector tests ---


class TestUnregisterConnector:
    """Tests for unregister_connector method."""

    async def test_unregister_existing_connector(self, connector_service, mock_kvstore):
        """Test unregistering an existing connector removes it from store."""
        # Register a connector
        await connector_service.register_connector(
            connector_id="to-remove",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
        )

        # Verify it exists
        assert await mock_kvstore.get(f"{KEY_PREFIX}to-remove") is not None

        # Unregister it
        await connector_service.unregister_connector("to-remove")

        # Verify it's gone
        assert await mock_kvstore.get(f"{KEY_PREFIX}to-remove") is None

    async def test_unregister_nonexistent_connector_does_not_raise(self, connector_service):
        """Test unregistering a non-existent connector doesn't raise an error."""
        # Should not raise - graceful handling
        await connector_service.unregister_connector("does-not-exist")


# --- Key prefix tests ---


class TestKeyPrefix:
    """Tests for connector key namespacing."""

    async def test_connectors_use_namespaced_keys(self, connector_service, mock_kvstore):
        """Test that connectors are stored with the correct key prefix."""
        await connector_service.register_connector(
            connector_id="test-connector",
            connector_type=ConnectorType.MCP,
            url="http://localhost:8080/mcp",
        )

        # Check the key was stored with prefix
        keys = list(mock_kvstore._storage.keys())
        assert len(keys) == 1
        assert keys[0] == "connectors:v1:test-connector"


# --- OpenAIResponseInputToolMCP validation tests ---


class TestMCPToolValidation:
    """Tests for MCP tool input validation."""

    def test_mcp_tool_requires_server_url_or_connector_id(self):
        """Test that either server_url or connector_id must be provided."""
        with pytest.raises(ValueError, match="server_url.*connector_id"):
            OpenAIResponseInputToolMCP(
                type="mcp",
                server_label="test",
                # Neither server_url nor connector_id provided
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

        # Call the actual helper function
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

        # Call the actual helper function
        resolved_tool = await resolve_mcp_connector_id(mcp_tool, mock_connectors_api)

        # Should keep original URL
        assert resolved_tool.server_url == "http://original-server:8080/mcp"
        # Should not call connectors API since server_url already exists
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
