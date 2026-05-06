# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.turn_events import StepCompleted, StepProgress, ToolCallIssuedDelta

AUTH_TOKEN = "test-token"

from tests.common.mcp import MCP_TOOLGROUP_ID, make_mcp_server


@pytest.fixture(scope="function")
def mcp_server():
    with make_mcp_server(required_auth_token=AUTH_TOKEN) as mcp_server_info:
        yield mcp_server_info


def test_mcp_invocation(ogx_client, text_model_id, mcp_server):
    """Test MCP tool invocation through the Responses/Agent API.

    MCP tools are passed directly as tool definitions to the Agent;
    the ToolGroups and ToolRuntime APIs are internal and not exposed over HTTP.
    """
    test_toolgroup_id = MCP_TOOLGROUP_ID
    uri = mcp_server["server_url"]

    print(f"Using model: {text_model_id}")
    tool_defs = [
        {
            "type": "mcp",
            "server_url": uri,
            "server_label": test_toolgroup_id,
            "require_approval": "never",
            "allowed_tools": ["greet_everyone", "get_boiling_point"],
            "authorization": AUTH_TOKEN,
        }
    ]
    agent = Agent(
        client=ogx_client,
        model=text_model_id,
        instructions="You are a helpful assistant.",
        tools=tool_defs,
    )
    session_id = agent.create_session("test-session")
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
                            "text": "Say hi to the world. Use tools to do so.",
                        }
                    ],
                }
            ],
            stream=True,
        )
    )
    events = [chunk.event for chunk in chunks]

    final_response = next((chunk.response for chunk in reversed(chunks) if chunk.response), None)
    assert final_response is not None

    issued_calls = [
        event for event in events if isinstance(event, StepProgress) and isinstance(event.delta, ToolCallIssuedDelta)
    ]
    assert issued_calls

    assert issued_calls[-1].delta.tool_name == "greet_everyone"

    tool_events = [
        event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
    ]
    assert tool_events
    assert tool_events[-1].result.tool_calls[0].tool_name == "greet_everyone"

    assert "hello" in final_response.output_text.lower()
