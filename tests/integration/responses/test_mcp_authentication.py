# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import llama_stack_client
import openai
import pytest

from tests.common.mcp import make_mcp_server

from .helpers import setup_mcp_tools

# MCP authentication tests with recordings
# Tests for bearer token authorization support in MCP tool configurations


def test_mcp_authorization_bearer(responses_client, text_model_id):
    """Test that bearer authorization is correctly applied to MCP requests."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    test_token = "test-bearer-token-789"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "auth-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "authorization": test_token,  # Just the token, not "Bearer <token>"
                }
            ],
            mcp_server_info,
        )

        # Create response - authorization should be applied
        response = responses_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify list_tools succeeded (requires auth)
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert len(response.output[0].tools) == 2

        # Verify tool invocation succeeded (requires auth)
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None


def test_mcp_authorization_error_when_header_provided(responses_client, text_model_id):
    """Test that providing Authorization in headers raises a security error."""
    test_token = "test-token-123"
    with make_mcp_server(required_auth_token=test_token) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "header-auth-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                    "headers": {"Authorization": f"Bearer {test_token}"},  # Security risk - should be rejected
                }
            ],
            mcp_server_info,
        )

        # Create response - should raise BadRequestError for security reasons
        with pytest.raises(
            (llama_stack_client.BadRequestError, openai.BadRequestError), match="'authorization' parameter"
        ):
            responses_client.responses.create(
                model=text_model_id,
                input="What is the boiling point of myawesomeliquid?",
                tools=tools,
                stream=False,
            )


def test_mcp_authorization_backward_compatibility(responses_client, text_model_id):
    """Test that MCP tools work without authorization (backward compatibility)."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    # No authorization required
    with make_mcp_server(required_auth_token=None) as mcp_server_info:
        tools = setup_mcp_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "noauth-mcp",
                    "server_url": "<FILLED_BY_TEST_RUNNER>",
                }
            ],
            mcp_server_info,
        )

        # Create response without authorization
        response = responses_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid?",
            tools=tools,
            stream=False,
        )

        # Verify operations succeeded without auth
        assert len(response.output) >= 3
        assert response.output[0].type == "mcp_list_tools"
        assert response.output[1].type == "mcp_call"
        assert response.output[1].error is None
