# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# we want the mcp server to be authenticated OR not, depends
from collections.abc import Callable
from contextlib import contextmanager

# Unfortunately the toolgroup id must be tied to the tool names because the registry
# indexes on both toolgroups and tools independently (and not jointly). That really
# needs to be fixed.
MCP_TOOLGROUP_ID = "mcp::localmcp"


def default_tools():
    """Default tools for backward compatibility."""
    from mcp.server.fastmcp import Context

    async def greet_everyone(url: str, ctx: Context) -> str:
        return "Hello, world!"

    async def get_boiling_point(liquid_name: str, celsius: bool = True) -> int:
        """
        Returns the boiling point of a liquid in Celsius or Fahrenheit.

        :param liquid_name: The name of the liquid
        :param celsius: Whether to return the boiling point in Celsius
        :return: The boiling point of the liquid in Celcius or Fahrenheit
        """
        if liquid_name.lower() == "myawesomeliquid":
            if celsius:
                return -100
            else:
                return -212
        else:
            return -1

    return {"greet_everyone": greet_everyone, "get_boiling_point": get_boiling_point}


def dependency_tools():
    """Tools with natural dependencies for multi-turn testing."""
    from mcp.server.fastmcp import Context

    async def get_user_id(username: str, ctx: Context) -> str:
        """
        Get the user ID for a given username. This ID is needed for other operations.

        :param username: The username to look up
        :return: The user ID for the username
        """
        # Simple mapping for testing
        user_mapping = {"alice": "user_12345", "bob": "user_67890", "charlie": "user_11111", "admin": "user_00000"}
        return user_mapping.get(username.lower(), "user_99999")

    async def get_user_permissions(user_id: str, ctx: Context) -> str:
        """
        Get the permissions for a user ID. Requires a valid user ID from get_user_id.

        :param user_id: The user ID to check permissions for
        :return: The permissions for the user
        """
        # Permission mapping based on user IDs
        permission_mapping = {
            "user_12345": "read,write",  # alice
            "user_67890": "read",  # bob
            "user_11111": "admin",  # charlie
            "user_00000": "superadmin",  # admin
            "user_99999": "none",  # unknown users
        }
        return permission_mapping.get(user_id, "none")

    async def check_file_access(user_id: str, filename: str, ctx: Context) -> str:
        """
        Check if a user can access a specific file. Requires a valid user ID.

        :param user_id: The user ID to check access for
        :param filename: The filename to check access to
        :return: Whether the user can access the file (yes/no)
        """
        # Get permissions first
        permission_mapping = {
            "user_12345": "read,write",  # alice
            "user_67890": "read",  # bob
            "user_11111": "admin",  # charlie
            "user_00000": "superadmin",  # admin
            "user_99999": "none",  # unknown users
        }
        permissions = permission_mapping.get(user_id, "none")

        # Check file access based on permissions and filename
        if permissions == "superadmin":
            access = "yes"
        elif permissions == "admin":
            access = "yes" if not filename.startswith("secret_") else "no"
        elif "write" in permissions:
            access = "yes" if filename.endswith(".txt") else "no"
        elif "read" in permissions:
            access = "yes" if filename.endswith(".txt") or filename.endswith(".md") else "no"
        else:
            access = "no"

        return access

    async def get_experiment_id(experiment_name: str, ctx: Context) -> str:
        """
        Get the experiment ID for a given experiment name. This ID is needed to get results.

        :param experiment_name: The name of the experiment
        :return: The experiment ID
        """
        # Simple mapping for testing
        experiment_mapping = {
            "temperature_test": "exp_001",
            "pressure_test": "exp_002",
            "chemical_reaction": "exp_003",
            "boiling_point": "exp_004",
        }
        exp_id = experiment_mapping.get(experiment_name.lower(), "exp_999")
        return exp_id

    async def get_experiment_results(experiment_id: str, ctx: Context) -> str:
        """
        Get the results for an experiment ID. Requires a valid experiment ID from get_experiment_id.

        :param experiment_id: The experiment ID to get results for
        :return: The experiment results
        """
        # Results mapping based on experiment IDs
        results_mapping = {
            "exp_001": "Temperature: 25°C, Status: Success",
            "exp_002": "Pressure: 1.2 atm, Status: Success",
            "exp_003": "Yield: 85%, Status: Complete",
            "exp_004": "Boiling Point: 100°C, Status: Verified",
            "exp_999": "No results found",
        }
        results = results_mapping.get(experiment_id, "Invalid experiment ID")
        return results

    return {
        "get_user_id": get_user_id,
        "get_user_permissions": get_user_permissions,
        "check_file_access": check_file_access,
        "get_experiment_id": get_experiment_id,
        "get_experiment_results": get_experiment_results,
    }


@contextmanager
def make_mcp_server(
    required_auth_token: str | None = None,
    tools: dict[str, Callable] | None = None,
    port: int | None = None,
):
    """
    Create an MCP server with the specified tools.

    :param required_auth_token: Optional auth token required for access
    :param tools: Dictionary of tool_name -> tool_function. If None, uses default tools.
    :param port: Optional fixed port to listen on. If None, an available port is chosen automatically.
    """
    import threading
    import time

    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    from ogx.log import get_logger

    server = FastMCP("FastMCP Test Server", log_level="WARNING")

    # Silence verbose MCP server logs
    import logging  # allow-direct-logging

    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

    tools = tools or default_tools()

    # Register all tools with the server
    for tool_func in tools.values():
        server.tool()(tool_func)

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        from starlette.exceptions import HTTPException

        auth_header: str | None = request.headers.get("Authorization")
        auth_token = None
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.split(" ")[1]

        if required_auth_token and auth_token != required_auth_token:
            raise HTTPException(status_code=401, detail="Unauthorized")

        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server._mcp_server.run(
                streams[0],
                streams[1],
                server._mcp_server.create_initialization_options(),
            )
            return Response()

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    def get_open_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    if port is None:
        port = get_open_port()
    logger = get_logger(__name__, category="tests::mcp")

    # make uvicorn logs be less verbose
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server_instance = uvicorn.Server(config)
    app.state.uvicorn_server = server_instance

    def run_server():
        try:
            logger.debug(f"Starting MCP server on port {port}")
            server_instance.run()
            logger.debug(f"MCP server on port {port} has stopped")
        except Exception as e:
            logger.error(f"MCP server failed to start on port {port}: {e}")
            raise

    # Start the server in a new thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    logger.debug(f"Starting MCP server thread on port {port}")
    server_thread.start()

    # Wait for the server to be actually listening on the port
    timeout = 5
    start_time = time.time()

    # Determine the appropriate host for the server URL based on test environment
    # - For library client and server mode: use localhost (both on same host)
    # - For docker mode: use host.docker.internal (container needs to reach host)
    import os

    mcp_host = os.environ.get("OGX_TEST_MCP_HOST", "localhost")
    server_url = f"http://{mcp_host}:{port}/sse"
    logger.debug(f"Waiting for MCP server to listen on port {port} (accessible via {mcp_host})")

    import socket

    server_ready = False
    while time.time() - start_time < timeout:
        if not server_thread.is_alive():
            logger.error(f"MCP server thread died unexpectedly on port {port}")
            break

        # Check if port is actually listening
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                result = sock.connect_ex(("127.0.0.1", port))
                if result == 0:
                    logger.debug(f"MCP server is ready on port {port}")
                    server_ready = True
                    break
        except Exception as e:
            logger.debug(f"Port check failed: {e}")

        time.sleep(0.1)

    if not server_ready:
        logger.error(f"MCP server failed to become ready within {timeout} seconds on port {port}")

    try:
        yield {"server_url": server_url}
    finally:
        logger.debug(f"Shutting down MCP server on port {port}")
        server_instance.should_exit = True
        time.sleep(0.5)

        # Force shutdown if still running
        if server_thread.is_alive():
            try:
                logger.debug("Force shutting down server thread")
                if hasattr(server_instance, "servers") and server_instance.servers:
                    for srv in server_instance.servers:
                        srv.close()

                # Wait for graceful shutdown
                server_thread.join(timeout=3)
                if server_thread.is_alive():
                    logger.warning("Server thread still alive after shutdown attempt")
            except Exception as e:
                logger.error(f"Error during server shutdown: {e}")

        # CRITICAL: Reset SSE global state to prevent event loop contamination
        # Reset the SSE AppStatus singleton that stores anyio.Event objects
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit = False
        AppStatus.should_exit_event = None
