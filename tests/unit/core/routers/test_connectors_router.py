# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

# Mark all async tests in this module to use anyio with asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


from ogx_api import (
    Connector,
    ConnectorNotFoundError,
    Connectors,
    ConnectorToolNotFoundError,
    ConnectorType,
    ListConnectorsResponse,
    ListToolsResponse,
    ToolDef,
)
from ogx_api.connectors.fastapi_routes import create_router
from ogx_api.connectors.models import (
    GetConnectorRequest,
    ListConnectorToolsRequest,
)


def _create_mock_connector():
    """Create a mock connector for testing."""
    return Connector(
        connector_id="test-connector",
        connector_type=ConnectorType.MCP,
        url="http://localhost:8080/mcp",
        server_name="Test Server",
        server_description="A test MCP server",
        server_version="1.0.0",
    )


def _create_mock_tool():
    """Create a mock tool definition for testing."""
    return ToolDef(
        name="test-tool",
        description="A test tool",
    )


def _get_endpoint(router, path: str, method: str = "GET"):
    """Get an endpoint function from router by path and method."""
    return next(
        r.endpoint for r in router.routes if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )


# --- List Connectors Tests ---


async def test_list_connectors_returns_empty_list():
    """Test listing connectors when none are registered."""
    impl = AsyncMock(spec=Connectors)
    impl.list_connectors.return_value = ListConnectorsResponse(data=[])

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1beta/connectors", "GET")
    response = await list_endpoint()

    assert response.data == []
    impl.list_connectors.assert_awaited_once()


async def test_list_connectors_returns_connectors():
    """Test listing connectors returns registered connectors."""
    impl = AsyncMock(spec=Connectors)
    mock_connector = _create_mock_connector()
    impl.list_connectors.return_value = ListConnectorsResponse(data=[mock_connector])

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1beta/connectors", "GET")
    response = await list_endpoint()

    assert len(response.data) == 1
    assert response.data[0].connector_id == "test-connector"
    assert response.data[0].connector_type == ConnectorType.MCP
    impl.list_connectors.assert_awaited_once()


# --- Get Connector Tests ---


async def test_get_connector_returns_connector():
    """Test getting a connector by ID."""
    impl = AsyncMock(spec=Connectors)
    mock_connector = _create_mock_connector()
    impl.get_connector.return_value = mock_connector

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}", "GET")
    request = GetConnectorRequest(connector_id="test-connector")
    response = await get_endpoint(request=request, authorization=None)

    assert response.connector_id == "test-connector"
    assert response.server_name == "Test Server"
    impl.get_connector.assert_awaited_once()


async def test_get_connector_with_authorization():
    """Test getting a connector with authorization token."""
    impl = AsyncMock(spec=Connectors)
    mock_connector = _create_mock_connector()
    impl.get_connector.return_value = mock_connector

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}", "GET")
    request = GetConnectorRequest(connector_id="test-connector")
    response = await get_endpoint(request=request, authorization="test-token")

    assert response.connector_id == "test-connector"
    # Verify authorization was passed to the impl
    call_args = impl.get_connector.call_args
    assert call_args.kwargs.get("authorization") == "test-token"


async def test_get_connector_not_found_raises_error():
    """Test getting a non-existent connector raises ConnectorNotFoundError."""
    impl = AsyncMock(spec=Connectors)
    impl.get_connector.side_effect = ConnectorNotFoundError("nonexistent")

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}", "GET")
    request = GetConnectorRequest(connector_id="nonexistent")

    with pytest.raises(ConnectorNotFoundError):
        await get_endpoint(request=request, authorization=None)


# --- List Connector Tools Tests ---


async def test_list_connector_tools_returns_tools():
    """Test listing tools from a connector."""
    impl = AsyncMock(spec=Connectors)
    mock_tool = _create_mock_tool()
    impl.list_connector_tools.return_value = ListToolsResponse(data=[mock_tool])

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}/tools", "GET")
    request = ListConnectorToolsRequest(connector_id="test-connector")
    response = await list_endpoint(request=request, authorization=None)

    assert len(response.data) == 1
    assert response.data[0].name == "test-tool"
    impl.list_connector_tools.assert_awaited_once()


async def test_list_connector_tools_empty():
    """Test listing tools when connector has no tools."""
    impl = AsyncMock(spec=Connectors)
    impl.list_connector_tools.return_value = ListToolsResponse(data=[])

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}/tools", "GET")
    request = ListConnectorToolsRequest(connector_id="test-connector")
    response = await list_endpoint(request=request, authorization=None)

    assert response.data == []


async def test_list_connector_tools_with_authorization():
    """Test listing tools with authorization token."""
    impl = AsyncMock(spec=Connectors)
    mock_tool = _create_mock_tool()
    impl.list_connector_tools.return_value = ListToolsResponse(data=[mock_tool])

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    list_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}/tools", "GET")
    request = ListConnectorToolsRequest(connector_id="test-connector")
    _ = await list_endpoint(request=request, authorization="test-token")

    call_args = impl.list_connector_tools.call_args
    assert call_args.kwargs.get("authorization") == "test-token"


# --- Get Connector Tool Tests ---


async def test_get_connector_tool_returns_tool():
    """Test getting a specific tool from a connector."""
    impl = AsyncMock(spec=Connectors)
    mock_tool = _create_mock_tool()
    impl.get_connector_tool.return_value = mock_tool

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}/tools/{tool_name}", "GET")
    response = await get_endpoint(
        connector_id="test-connector",
        tool_name="test-tool",
        authorization=None,
    )

    assert response.name == "test-tool"
    assert response.description == "A test tool"
    impl.get_connector_tool.assert_awaited_once()


async def test_get_connector_tool_with_authorization():
    """Test getting a tool with authorization token."""
    impl = AsyncMock(spec=Connectors)
    mock_tool = _create_mock_tool()
    impl.get_connector_tool.return_value = mock_tool

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}/tools/{tool_name}", "GET")
    _ = await get_endpoint(
        connector_id="test-connector",
        tool_name="test-tool",
        authorization="test-token",
    )

    call_args = impl.get_connector_tool.call_args
    assert call_args.kwargs.get("authorization") == "test-token"


async def test_get_connector_tool_not_found_raises_error():
    """Test getting a non-existent tool raises ConnectorToolNotFoundError."""
    impl = AsyncMock(spec=Connectors)
    impl.get_connector_tool.side_effect = ConnectorToolNotFoundError("test-connector", "nonexistent-tool")

    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    get_endpoint = _get_endpoint(router, "/v1beta/connectors/{connector_id}/tools/{tool_name}", "GET")

    with pytest.raises(ConnectorToolNotFoundError):
        await get_endpoint(
            connector_id="test-connector",
            tool_name="nonexistent-tool",
            authorization=None,
        )


# --- OpenAPI Schema Tests ---


def test_openapi_schema_has_connectors_endpoints():
    """Test that OpenAPI schema includes all connectors endpoints."""
    impl = AsyncMock(spec=Connectors)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()

    # Verify all endpoints are documented
    assert "/v1beta/connectors" in schema["paths"]
    assert "/v1beta/connectors/{connector_id}" in schema["paths"]
    assert "/v1beta/connectors/{connector_id}/tools" in schema["paths"]
    assert "/v1beta/connectors/{connector_id}/tools/{tool_name}" in schema["paths"]


def test_openapi_schema_list_connectors_is_get():
    """Test list connectors endpoint is documented as GET."""
    impl = AsyncMock(spec=Connectors)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()
    connectors_path = schema["paths"]["/v1beta/connectors"]

    assert "get" in connectors_path
    assert connectors_path["get"]["summary"] == "List all connectors."


def test_openapi_schema_get_connector_has_path_param():
    """Test get connector endpoint has connector_id path parameter."""
    impl = AsyncMock(spec=Connectors)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()
    get_connector_path = schema["paths"]["/v1beta/connectors/{connector_id}"]

    assert "get" in get_connector_path
    parameters = get_connector_path["get"]["parameters"]
    param_names = [p["name"] for p in parameters]
    assert "connector_id" in param_names


def test_openapi_schema_has_authorization_query_param():
    """Test that endpoints have authorization query parameter."""
    impl = AsyncMock(spec=Connectors)
    app = FastAPI()
    router = create_router(impl)
    app.include_router(router)

    schema = app.openapi()
    get_connector_path = schema["paths"]["/v1beta/connectors/{connector_id}"]

    parameters = get_connector_path["get"]["parameters"]
    auth_params = [p for p in parameters if p["name"] == "authorization"]
    assert len(auth_params) == 1
    assert auth_params[0]["in"] == "query"
