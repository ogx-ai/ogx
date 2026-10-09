# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import httpx2
from mcp import ClientSession, MCPError
from mcp import types as mcp_types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from ogx.core.datatypes import AuthenticationRequiredError
from ogx.log import get_logger
from ogx.providers.utils.tools.ttl_dict import TTLDict
from ogx_api import (
    ImageContentItem,
    InterleavedContentItem,
    ListToolDefsResponse,
    TextContentItem,
    ToolDef,
    ToolInvocationResult,
    _URLOrData,
)

logger = get_logger(__name__, category="tools")


def prepare_mcp_headers(base_headers: dict[str, str] | None, authorization: str | None) -> dict[str, str]:
    """
    Prepare headers for MCP requests with authorization support.

    Args:
        base_headers: Base headers dictionary (can be None)
        authorization: OAuth access token (without "Bearer " prefix)

    Returns:
        Headers dictionary with Authorization header if token provided

    Raises:
        ValueError: If Authorization header is specified in the headers dict (security risk)
    """
    headers = dict(base_headers or {})

    # Security check: reject any Authorization header in the headers dict
    # Users must use the authorization parameter instead to avoid security risks
    existing_keys_lower = {k.lower() for k in headers.keys()}
    if "authorization" in existing_keys_lower:
        raise ValueError(
            "For security reasons, Authorization header cannot be passed via 'headers'. "
            "Please use the 'authorization' parameter instead."
        )

    # Add Authorization header if token provided
    if authorization:
        # OAuth access token - add "Bearer " prefix
        headers["Authorization"] = f"Bearer {authorization}"

    return headers


protocol_cache = TTLDict(ttl_seconds=3600)


class MCPProtol(Enum):
    """Enumeration of supported MCP transport protocols."""

    UNKNOWN = 0
    STREAMABLE_HTTP = 1
    SSE = 2


class MCPSessionManager:
    """Manages MCP session lifecycle within a request scope.

    This class caches MCP sessions by (endpoint, headers_hash) to avoid redundant
    connection establishment and tools/list calls when making multiple tool
    invocations to the same MCP server within a single request.

    Fix for GitHub issue #4452: MCP tools/list called redundantly before every
    tool invocation.

    Usage:
        async with MCPSessionManager() as session_manager:
            # Multiple tool calls will reuse the same session
            result1 = await invoke_mcp_tool(..., session_manager=session_manager)
            result2 = await invoke_mcp_tool(..., session_manager=session_manager)
    """

    def __init__(self):
        # Cache of active sessions: key -> (session, client_context, session_context)
        self._sessions: dict[str, tuple[ClientSession, Any, Any]] = {}
        # Locks to prevent concurrent session creation for the same key
        self._locks: dict[str, asyncio.Lock] = {}
        # Global lock for managing the locks dict
        self._global_lock = asyncio.Lock()

    def _make_key(self, endpoint: str, headers: dict[str, str]) -> str:
        """Create a cache key from endpoint and headers."""
        # Sort headers for consistent hashing
        headers_str = str(sorted(headers.items()))
        headers_hash = hashlib.sha256(headers_str.encode()).hexdigest()[:16]
        return f"{endpoint}:{headers_hash}"

    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a specific cache key."""
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def get_session(self, endpoint: str, headers: dict[str, str]) -> ClientSession:
        """Get or create an MCP session for the given endpoint and headers.

        Args:
            endpoint: MCP server endpoint URL
            headers: Headers including authorization

        Returns:
            An initialized ClientSession ready for tool calls
        """
        key = self._make_key(endpoint, headers)

        # Check if session already exists (fast path)
        if key in self._sessions:
            session, _, _ = self._sessions[key]
            return session

        # Acquire lock for this specific key to prevent concurrent creation
        lock = await self._get_lock(key)
        async with lock:
            # Double-check after acquiring lock
            if key in self._sessions:
                session, _, _ = self._sessions[key]
                return session

            # Create new session
            session, client_ctx, session_ctx = await self._create_session(endpoint, headers)
            self._sessions[key] = (session, client_ctx, session_ctx)
            logger.debug("Created new MCP session", endpoint=endpoint, key=key[:32])
            return session

    async def _create_session(self, endpoint: str, headers: dict[str, str]) -> tuple[ClientSession, Any, Any]:
        """Create a new MCP session.

        Returns:
            Tuple of (session, client_context, session_context) for lifecycle management
        """
        # Use the same protocol detection logic as client_wrapper
        connection_strategies = [MCPProtol.STREAMABLE_HTTP, MCPProtol.SSE]
        mcp_protocol = protocol_cache.get(endpoint, default=MCPProtol.UNKNOWN)
        if mcp_protocol == MCPProtol.SSE:
            connection_strategies = [MCPProtol.SSE, MCPProtol.STREAMABLE_HTTP]

        last_exception: BaseException | None = None

        for i, strategy in enumerate(connection_strategies):
            try:
                # Enter the client context managers manually. Streamable HTTP takes
                # headers via an httpx2 client (caller-owned, so tracked on the exit
                # stack); SSE still accepts headers directly.
                client_stack = AsyncExitStack()
                try:
                    if strategy == MCPProtol.SSE:
                        transport = sse_client(endpoint, headers=headers)
                    else:
                        http_client = await client_stack.enter_async_context(httpx2.AsyncClient(headers=headers))
                        transport = streamable_http_client(endpoint, http_client=http_client)
                    client_streams = await client_stack.enter_async_context(transport)
                except httpx2.HTTPStatusError as e:
                    # A 401 during connect is raised outside the SDK's task group, so
                    # it bypasses the except* clauses below; map it explicitly.
                    await client_stack.__aexit__(None, None, None)
                    if e.response.status_code == 401:
                        raise AuthenticationRequiredError(e) from e
                    raise
                except BaseException:
                    await client_stack.__aexit__(None, None, None)
                    raise
                client_ctx = client_stack

                try:
                    # Enter the session context manager manually
                    session = ClientSession(read_stream=client_streams[0], write_stream=client_streams[1])
                    session_ctx = session
                    await session.__aenter__()

                    try:
                        await session.initialize()
                        protocol_cache[endpoint] = strategy
                        return session, client_ctx, session_ctx
                    except BaseException:
                        await session.__aexit__(None, None, None)
                        raise
                except BaseException:
                    await client_ctx.__aexit__(None, None, None)
                    raise

            except* httpx2.HTTPStatusError as eg:
                for exc in eg.exceptions:
                    err = cast(httpx2.HTTPStatusError, exc)
                    if err.response.status_code == 401:
                        raise AuthenticationRequiredError(exc) from exc
                if i == len(connection_strategies) - 1:
                    raise
                last_exception = eg
            except* httpx2.ConnectError as eg:
                if i == len(connection_strategies) - 1:
                    error_msg = f"Failed to connect to MCP server at {endpoint}: Connection refused"
                    logger.error("MCP connection error", error=error_msg)
                    raise ConnectionError(error_msg) from eg
                else:
                    logger.warning(
                        "failed to connect to MCP server at via , falling back to",
                        endpoint=endpoint,
                        name=strategy.name,
                        connection_strategies_i_1_name=connection_strategies[i + 1].name,
                    )
                last_exception = eg
            except* httpx2.TimeoutException as eg:
                if i == len(connection_strategies) - 1:
                    error_msg = f"MCP server at {endpoint} timed out"
                    logger.error("MCP timeout error", error=error_msg)
                    raise TimeoutError(error_msg) from eg
                else:
                    logger.warning(
                        "MCP server at timed out via , falling back to",
                        endpoint=endpoint,
                        name=strategy.name,
                        connection_strategies_i_1_name=connection_strategies[i + 1].name,
                    )
                last_exception = eg
            except* httpx2.RequestError as eg:
                if i == len(connection_strategies) - 1:
                    exc_msg = str(eg.exceptions[0]) if eg.exceptions else "Unknown error"
                    error_msg = f"Network error connecting to MCP server at {endpoint}: {exc_msg}"
                    logger.error("MCP network error", error=error_msg)
                    raise ConnectionError(error_msg) from eg
                else:
                    logger.warning(
                        "network error connecting to MCP server at via , falling back to",
                        endpoint=endpoint,
                        name=strategy.name,
                        connection_strategies_i_1_name=connection_strategies[i + 1].name,
                    )
                last_exception = eg
            except* MCPError:
                if i < len(connection_strategies) - 1:
                    logger.warning(
                        "failed to connect via , falling back to",
                        name=strategy.name,
                        connection_strategies_i_1_name=connection_strategies[i + 1].name,
                    )
                else:
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Failed to create MCP session for {endpoint}")

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager and cleanup all sessions.

        Note: We catch BaseException (not just Exception) because:
        1. CancelledError is a BaseException and can occur during cleanup
        2. anyio cancel scope errors can occur if cleanup runs in a different
           task context than where the session was created
        These are expected in streaming response scenarios and are handled gracefully.
        """
        errors = []
        session_count = len(self._sessions)
        for key, (session, client_ctx, _) in list(self._sessions.items()):
            try:
                await session.__aexit__(None, None, None)
            except BaseException as e:
                # Debug level since these errors are expected in streaming scenarios
                # where cleanup runs in a different async context than session creation
                logger.debug("Error closing MCP session", key=key, error=str(e))
                errors.append(e)
            try:
                await client_ctx.__aexit__(None, None, None)
            except BaseException as e:
                logger.debug("Error closing MCP client context", key=key, error=str(e))
                errors.append(e)

        self._sessions.clear()
        self._locks.clear()
        logger.debug("Closed MCP sessions", session_count=session_count)

        if errors:
            logger.debug(
                "Encountered errors while closing MCP sessions (expected in streaming)", errors_count=len(errors)
            )

        return False


@asynccontextmanager
async def client_wrapper(endpoint: str, headers: dict[str, str]) -> AsyncGenerator[ClientSession, Any]:
    """Create an MCP client session for the given endpoint with protocol auto-detection.

    Args:
        endpoint: MCP server endpoint URL
        headers: HTTP headers to include in the connection

    Returns:
        An async context manager yielding an initialized ClientSession
    """
    # we use a ttl'd dict to cache the happy path protocol for each endpoint
    # but, we always fall back to trying the other protocol if we cannot initialize the session
    connection_strategies = [MCPProtol.STREAMABLE_HTTP, MCPProtol.SSE]
    mcp_protocol = protocol_cache.get(endpoint, default=MCPProtol.UNKNOWN)
    if mcp_protocol == MCPProtol.SSE:
        connection_strategies = [MCPProtol.SSE, MCPProtol.STREAMABLE_HTTP]

    for i, strategy in enumerate(connection_strategies):
        try:
            # Streamable HTTP takes headers via an httpx2 client; SSE accepts them directly.
            async with AsyncExitStack() as client_stack:
                if strategy == MCPProtol.SSE:
                    transport = sse_client(endpoint, headers=headers)
                else:
                    http_client = await client_stack.enter_async_context(httpx2.AsyncClient(headers=headers))
                    transport = streamable_http_client(endpoint, http_client=http_client)

                # A 401 during connect is raised outside the SDK's task group, so it
                # bypasses the except* clauses below; map it explicitly.
                try:
                    async with transport as client_streams:
                        async with ClientSession(
                            read_stream=client_streams[0], write_stream=client_streams[1]
                        ) as session:
                            await session.initialize()
                            protocol_cache[endpoint] = strategy
                            yield session
                            return
                except httpx2.HTTPStatusError as e:
                    if e.response.status_code == 401:
                        raise AuthenticationRequiredError(e) from e
                    raise
        except* httpx2.HTTPStatusError as eg:
            for exc in eg.exceptions:
                # mypy does not currently narrow the type of `eg.exceptions` based on the `except*` filter,
                # so we explicitly cast each item to httpx2.HTTPStatusError. This is safe because
                # `except* httpx2.HTTPStatusError` guarantees all exceptions in `eg.exceptions` are of that type.
                err = cast(httpx2.HTTPStatusError, exc)
                if err.response.status_code == 401:
                    raise AuthenticationRequiredError(exc) from exc
            if i == len(connection_strategies) - 1:
                raise
        except* httpx2.ConnectError as eg:
            # Connection refused, server down, network unreachable
            if i == len(connection_strategies) - 1:
                error_msg = f"Failed to connect to MCP server at {endpoint}: Connection refused"
                logger.error("MCP connection error", error=error_msg)
                raise ConnectionError(error_msg) from eg
            else:
                logger.warning(
                    "failed to connect to MCP server at via , falling back to",
                    endpoint=endpoint,
                    name=strategy.name,
                    connection_strategies_i_1_name=connection_strategies[i + 1].name,
                )
        except* httpx2.TimeoutException as eg:
            # Request timeout, server too slow
            if i == len(connection_strategies) - 1:
                error_msg = f"MCP server at {endpoint} timed out"
                logger.error("MCP timeout error", error=error_msg)
                raise TimeoutError(error_msg) from eg
            else:
                logger.warning(
                    "MCP server at timed out via , falling back to",
                    endpoint=endpoint,
                    name=strategy.name,
                    connection_strategies_i_1_name=connection_strategies[i + 1].name,
                )
        except* httpx2.RequestError as eg:
            # DNS resolution failures, network errors, invalid URLs
            if i == len(connection_strategies) - 1:
                # Get the first exception's message for the error string
                exc_msg = str(eg.exceptions[0]) if eg.exceptions else "Unknown error"
                error_msg = f"Network error connecting to MCP server at {endpoint}: {exc_msg}"
                logger.error("MCP network error", error=error_msg)
                raise ConnectionError(error_msg) from eg
            else:
                logger.warning(
                    "network error connecting to MCP server at via , falling back to",
                    endpoint=endpoint,
                    name=strategy.name,
                    connection_strategies_i_1_name=connection_strategies[i + 1].name,
                )
        except* MCPError:
            if i < len(connection_strategies) - 1:
                logger.warning(
                    "failed to connect via , falling back to",
                    name=strategy.name,
                    connection_strategies_i_1_name=connection_strategies[i + 1].name,
                )
            else:
                raise


async def list_mcp_tools(
    endpoint: str,
    headers: dict[str, str] | None = None,
    authorization: str | None = None,
    session_manager: MCPSessionManager | None = None,
) -> ListToolDefsResponse:
    """List tools available from an MCP server.

    Args:
        endpoint: MCP server endpoint URL
        headers: Optional base headers to include
        authorization: Optional OAuth access token (just the token, not "Bearer <token>")
        session_manager: Optional MCPSessionManager for session reuse within a request.
            When provided, sessions are cached and reused, avoiding redundant session
            creation when list_mcp_tools and invoke_mcp_tool are called for the same
            server within a request. (Fix for #4452)

    Returns:
        List of tool definitions from the MCP server

    Raises:
        ValueError: If Authorization is found in the headers parameter
    """
    # Prepare headers with authorization handling
    final_headers = prepare_mcp_headers(headers, authorization)

    tools = []

    # Helper function to process session and list tools
    async def _list_tools_from_session(session):
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tools.append(
                ToolDef(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                    metadata={
                        "endpoint": endpoint,
                    },
                )
            )

    # If a session manager is provided, use it for session reuse (fix for #4452)
    if session_manager is not None:
        session = await session_manager.get_session(endpoint, final_headers)
        await _list_tools_from_session(session)
    else:
        # Fallback to original behavior: create a new session for this call
        async with client_wrapper(endpoint, final_headers) as session:
            await _list_tools_from_session(session)

    return ListToolDefsResponse(data=tools)


def _parse_mcp_result(result: mcp_types.CallToolResult) -> ToolInvocationResult:
    """Parse MCP tool call result into ToolInvocationResult.

    Every content block becomes something the model can read: text and images
    pass through, embedded text resources contribute their text, and binary
    resources, resource links and audio are described in a text item instead
    of failing the whole call. A result that only carries ``structured_content``
    is serialized to JSON so the model does not receive an empty tool message;
    the structured value is also kept under ``metadata["structured_content"]``.
    When the tool reports ``is_error``, its text becomes ``error_message``.

    Args:
        result: The raw MCP tool call result

    Returns:
        ToolInvocationResult with parsed content
    """
    content: list[InterleavedContentItem] = []
    for item in result.content:
        if isinstance(item, mcp_types.TextContent):
            content.append(TextContentItem(text=item.text))
        elif isinstance(item, mcp_types.ImageContent):
            content.append(ImageContentItem(image=_URLOrData(data=item.data)))
        elif isinstance(item, mcp_types.EmbeddedResource):
            resource = item.resource
            if isinstance(resource, mcp_types.TextResourceContents):
                content.append(TextContentItem(text=resource.text))
            else:
                mime_type = resource.mime_type or "unknown type"
                content.append(TextContentItem(text=f"[binary resource {resource.uri}, {mime_type}]"))
        elif isinstance(item, mcp_types.ResourceLink):
            description = f": {item.description}" if item.description else ""
            content.append(TextContentItem(text=f"[resource link {item.uri} ({item.name}){description}]"))
        elif isinstance(item, mcp_types.AudioContent):
            content.append(TextContentItem(text=f"[audio content, {item.mime_type}]"))
        else:
            # Keep the call alive and let the model see that something was there.
            logger.warning("Unsupported MCP content block", block_type=type(item).__name__)
            content.append(TextContentItem(text=f"[unsupported MCP content block: {type(item).__name__}]"))

    metadata = None
    if result.structured_content is not None:
        metadata = {"structured_content": result.structured_content}
        if not content:
            content.append(TextContentItem(text=json.dumps(result.structured_content)))

    error_message = None
    if result.is_error:
        error_message = "\n".join(item.text for item in content if isinstance(item, TextContentItem)) or None

    return ToolInvocationResult(
        content=content,
        error_code=1 if result.is_error else 0,
        error_message=error_message,
        metadata=metadata,
    )


async def invoke_mcp_tool(
    endpoint: str,
    tool_name: str,
    kwargs: dict[str, Any],
    headers: dict[str, str] | None = None,
    authorization: str | None = None,
    session_manager: MCPSessionManager | None = None,
) -> ToolInvocationResult:
    """Invoke an MCP tool with the given arguments.

    Args:
        endpoint: MCP server endpoint URL
        tool_name: Name of the tool to invoke
        kwargs: Tool invocation arguments
        headers: Optional base headers to include
        authorization: Optional OAuth access token (just the token, not "Bearer <token>")
        session_manager: Optional MCPSessionManager for session reuse within a request.
            When provided, sessions are cached and reused for multiple tool calls to
            the same endpoint, avoiding redundant tools/list calls. (Fix for #4452)

    Returns:
        Tool invocation result with content and error information

    Raises:
        ValueError: If Authorization header is found in the headers parameter
    """
    # Prepare headers with authorization handling
    final_headers = prepare_mcp_headers(headers, authorization)

    # If a session manager is provided, use it for session reuse (fix for #4452)
    if session_manager is not None:
        session = await session_manager.get_session(endpoint, final_headers)
        result = await session.call_tool(tool_name, kwargs)
        return _parse_mcp_result(result)

    # Fallback to original behavior: create a new session for each call
    async with client_wrapper(endpoint, final_headers) as session:
        result = await session.call_tool(tool_name, kwargs)
        return _parse_mcp_result(result)


@dataclass
class MCPServerInfo:
    """Server information from an MCP server."""

    name: str
    version: str
    title: str | None = None
    description: str | None = None


async def get_mcp_server_info(
    endpoint: str,
    headers: dict[str, str] | None = None,
    authorization: str | None = None,
) -> MCPServerInfo:
    """Get server info from an MCP server.
    Args:
        endpoint: MCP server endpoint URL
        headers: Optional base headers to include
        authorization: Optional OAuth access token (just the token, not "Bearer <token>")
    Returns:
        MCPServerInfo containing name, version, title, and description
    """
    final_headers = prepare_mcp_headers(headers, authorization)

    async with client_wrapper(endpoint, final_headers) as session:
        init_result = await session.initialize()

        return MCPServerInfo(
            name=init_result.server_info.name,
            version=init_result.server_info.version,
            title=init_result.server_info.title,
            description=init_result.instructions,
        )
