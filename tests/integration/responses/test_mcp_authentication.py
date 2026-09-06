# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import ogx_open_client
import openai
import pytest

from tests.common.mcp import make_mcp_server

from .helpers import skip_if_provider_is_vertexai

# MCP authentication tests with recordings
# Tests for bearer token authorization support in MCP tool configurations
CONNECTOR_MCP_PORT = 5199


def test_mcp_authorization_bearer(responses_client, client_with_models, text_model_id):
    """Test that bearer authorization is correctly applied to MCP requests."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "MCP tool calling behavior differs from expected output structure"
    )
    test_token = "test-bearer-token-789"
    with make_mcp_server(port=CONNECTOR_MCP_PORT, required_auth_token=test_token):
        tools = [
            {
                "type": "mcp",
                "server_label": "auth-mcp",
                "connector_id": "test-mcp-connector",
                "authorization": test_token,  # Just the token, not "Bearer <token>"
            }
        ]

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


def test_mcp_authorization_error_when_header_provided(responses_client, client_with_models, text_model_id):
    """Test that providing Authorization in headers raises a security error."""
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "MCP tool calling behavior differs from expected output structure"
    )
    test_token = "test-token-123"
    with make_mcp_server(port=CONNECTOR_MCP_PORT, required_auth_token=test_token):
        tools = [
            {
                "type": "mcp",
                "server_label": "header-auth-mcp",
                "connector_id": "test-mcp-connector",
                "headers": {"Authorization": f"Bearer {test_token}"},  # Security risk - should be rejected
            }
        ]

        # Create response - should raise BadRequestError for security reasons
        with pytest.raises(
            (ogx_open_client.BadRequestError, openai.BadRequestError), match="'authorization' parameter"
        ):
            responses_client.responses.create(
                model=text_model_id,
                input="What is the boiling point of myawesomeliquid?",
                tools=tools,
                stream=False,
            )


def test_mcp_authorization_backward_compatibility(responses_client, client_with_models, text_model_id):
    """Test that MCP tools work without authorization (backward compatibility)."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "MCP tool calling behavior differs from expected output structure"
    )
    # No authorization required
    with make_mcp_server(port=CONNECTOR_MCP_PORT, required_auth_token=None):
        tools = [
            {
                "type": "mcp",
                "server_label": "noauth-mcp",
                "connector_id": "test-mcp-connector",
            }
        ]

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
