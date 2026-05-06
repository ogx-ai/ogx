# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for MCP tools with complex JSON Schema support.
Tests $ref, $defs, and other JSON Schema features through MCP integration
via the Responses/Agent API.
"""

import pytest
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.turn_events import StepCompleted, StepProgress, ToolCallIssuedDelta

from tests.common.mcp import make_mcp_server

AUTH_TOKEN = "test-token"


@pytest.fixture(scope="function")
def mcp_server_with_complex_schemas():
    """MCP server with tools that have complex schemas including $ref and $defs."""
    from mcp.server.fastmcp import Context

    async def book_flight(flight: dict, passengers: list[dict], payment: dict, ctx: Context) -> dict:
        """
        Book a flight with passenger and payment information.

        This tool uses JSON Schema $ref and $defs for type reuse.
        """
        return {
            "booking_id": "BK12345",
            "flight": flight,
            "passengers": passengers,
            "payment": payment,
            "status": "confirmed",
        }

    async def process_order(order_data: dict, ctx: Context) -> dict:
        """
        Process an order with nested address information.

        Uses nested objects and $ref.
        """
        return {"order_id": "ORD789", "status": "processing", "data": order_data}

    async def flexible_contact(contact_info: str, ctx: Context) -> dict:
        """
        Accept flexible contact (email or phone).

        Uses anyOf schema.
        """
        if "@" in contact_info:
            return {"type": "email", "value": contact_info}
        else:
            return {"type": "phone", "value": contact_info}

    tools = {"book_flight": book_flight, "process_order": process_order, "flexible_contact": flexible_contact}

    with make_mcp_server(required_auth_token=AUTH_TOKEN, tools=tools) as server_info:
        yield server_info


@pytest.fixture(scope="function")
def mcp_server_with_output_schemas():
    """MCP server with tools that have output schemas defined."""
    from mcp.server.fastmcp import Context

    async def get_weather(location: str, ctx: Context) -> dict:
        """
        Get weather with structured output.

        Has both input and output schemas.
        """
        return {"temperature": 72.5, "conditions": "Sunny", "humidity": 45, "wind_speed": 10.2}

    async def calculate(x: float, y: float, operation: str, ctx: Context) -> dict:
        """
        Perform calculation with validated output.
        """
        operations = {"add": x + y, "subtract": x - y, "multiply": x * y, "divide": x / y if y != 0 else None}
        result = operations.get(operation)
        return {"result": result, "operation": operation}

    tools = {"get_weather": get_weather, "calculate": calculate}

    with make_mcp_server(required_auth_token=AUTH_TOKEN, tools=tools) as server_info:
        yield server_info


@pytest.mark.skip(reason="Requires a capable model; small models loop indefinitely on tool selection")
class TestMCPToolInvocationViaAgent:
    """Test invoking MCP tools with complex schemas through the Agent API."""

    def test_invoke_mcp_tool_with_nested_data(self, ogx_client, text_model_id, mcp_server_with_complex_schemas):
        """Test that an MCP tool accepting nested object structures can be invoked via the Agent."""
        uri = mcp_server_with_complex_schemas["server_url"]

        tool_defs = [
            {
                "type": "mcp",
                "server_url": uri,
                "server_label": "complex-schemas",
                "require_approval": "never",
                "allowed_tools": ["process_order"],
                "authorization": AUTH_TOKEN,
            }
        ]

        agent = Agent(
            client=ogx_client,
            model=text_model_id,
            instructions="You are a helpful assistant. When asked to process an order, use the process_order tool.",
            tools=tool_defs,
        )
        session_id = agent.create_session("test-nested-data")
        chunks = list(
            agent.create_turn(
                session_id=session_id,
                messages=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Process an order with 2 widgets and 1 gadget, "
                                    "shipping to 123 Main St, San Francisco 94102."
                                ),
                            }
                        ],
                    }
                ],
                stream=True,
            )
        )

        events = [chunk.event for chunk in chunks]

        issued_calls = [
            event
            for event in events
            if isinstance(event, StepProgress) and isinstance(event.delta, ToolCallIssuedDelta)
        ]
        assert issued_calls, "Expected at least one tool call to be issued"
        assert issued_calls[-1].delta.tool_name == "process_order"

        tool_events = [
            event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
        ]
        assert tool_events, "Expected tool execution to complete"
        assert tool_events[-1].result.tool_calls[0].tool_name == "process_order"

    def test_invoke_with_flexible_schema(self, ogx_client, text_model_id, mcp_server_with_complex_schemas):
        """Test invoking a tool that accepts flexible input (email or phone)."""
        uri = mcp_server_with_complex_schemas["server_url"]

        tool_defs = [
            {
                "type": "mcp",
                "server_url": uri,
                "server_label": "complex-schemas",
                "require_approval": "never",
                "allowed_tools": ["flexible_contact"],
                "authorization": AUTH_TOKEN,
            }
        ]

        agent = Agent(
            client=ogx_client,
            model=text_model_id,
            instructions="You are a helpful assistant. Use the flexible_contact tool when given contact info.",
            tools=tool_defs,
        )
        session_id = agent.create_session("test-flexible")
        chunks = list(
            agent.create_turn(
                session_id=session_id,
                messages=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Save this contact info: user@example.com",
                            }
                        ],
                    }
                ],
                stream=True,
            )
        )

        events = [chunk.event for chunk in chunks]
        tool_events = [
            event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
        ]
        assert tool_events, "Expected tool execution to complete"
        assert tool_events[-1].result.tool_calls[0].tool_name == "flexible_contact"


@pytest.mark.skip(reason="Requires a capable model; small models loop indefinitely on tool selection")
class TestMCPSchemaRoundTrip:
    """Test that MCP tool schemas survive the full round-trip:
    MCP server -> tool discovery -> LLM tool call -> MCP invocation -> result.
    """

    def test_complex_tool_produces_valid_result(self, ogx_client, text_model_id, mcp_server_with_output_schemas):
        """Test that a tool with structured output returns valid data through the Agent."""
        uri = mcp_server_with_output_schemas["server_url"]

        tool_defs = [
            {
                "type": "mcp",
                "server_url": uri,
                "server_label": "output-schemas",
                "require_approval": "never",
                "allowed_tools": ["calculate"],
                "authorization": AUTH_TOKEN,
            }
        ]

        agent = Agent(
            client=ogx_client,
            model=text_model_id,
            instructions="You are a helpful calculator. Use the calculate tool to answer math questions.",
            tools=tool_defs,
        )
        session_id = agent.create_session("test-output-schema")
        chunks = list(
            agent.create_turn(
                session_id=session_id,
                messages=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "What is 7 + 3? Use the calculate tool with operation 'add'.",
                            }
                        ],
                    }
                ],
                stream=True,
            )
        )

        events = [chunk.event for chunk in chunks]

        tool_events = [
            event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
        ]
        assert tool_events, "Expected tool execution to complete"

        tool_result = tool_events[-1].result.tool_calls[0]
        assert tool_result.tool_name == "calculate"

        final_response = next((chunk.response for chunk in reversed(chunks) if chunk.response), None)
        assert final_response is not None
        assert "10" in final_response.output_text

    def test_multi_tool_mcp_server(self, ogx_client, text_model_id, mcp_server_with_complex_schemas):
        """Test that multiple tools from the same MCP server are all discoverable and callable."""
        uri = mcp_server_with_complex_schemas["server_url"]

        tool_defs = [
            {
                "type": "mcp",
                "server_url": uri,
                "server_label": "complex-schemas",
                "require_approval": "never",
                "allowed_tools": ["book_flight", "process_order", "flexible_contact"],
                "authorization": AUTH_TOKEN,
            }
        ]

        agent = Agent(
            client=ogx_client,
            model=text_model_id,
            instructions=("You are a helpful assistant. Use the flexible_contact tool to save contact information."),
            tools=tool_defs,
        )
        session_id = agent.create_session("test-multi-tool")
        chunks = list(
            agent.create_turn(
                session_id=session_id,
                messages=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Save the contact info for phone number +15551234567.",
                            }
                        ],
                    }
                ],
                stream=True,
            )
        )

        events = [chunk.event for chunk in chunks]
        tool_events = [
            event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
        ]
        assert tool_events, "Expected tool execution to complete"

        final_response = next((chunk.response for chunk in reversed(chunks) if chunk.response), None)
        assert final_response is not None
