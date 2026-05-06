# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test that models can be routed using provider_id/model_id format
when the provider is configured but the specific model is not registered.

This test validates the fix in src/ogx/core/routers/inference.py
that enables routing based on provider_data alone.
"""

from unittest.mock import AsyncMock, patch

import pytest

from ogx.core.library_client import OGXAsLibraryClient
from ogx_api import (
    Api,
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
)


def test_unregistered_model_routing_with_provider_data(client_with_models):
    """
    Test that a model can be routed using provider_id/model_id format
    even when the model is not explicitly registered, as long as the provider
    is available.

    This validates the fix where the router:
    1. Tries to lookup model in routing table
    2. If not found, splits model_id by "/" to extract provider_id and provider_resource_id
    3. Routes directly to the provider with the provider_resource_id

    Without the fix, this would raise ModelNotFoundError immediately.
    With the fix, the routing succeeds and the request reaches the provider.
    """
    if not isinstance(client_with_models, OGXAsLibraryClient):
        pytest.skip("Test requires library client for provider-level patching")

    client = client_with_models

    # Use a model format that follows provider_id/model_id convention
    # We'll use anthropic as an example since it's a remote provider that
    # benefits from this pattern
    test_model_id = "anthropic/claude-3-5-sonnet-20241022"

    # First, verify the model is NOT registered
    registered_models = {m.id for m in client.models.list()}
    assert test_model_id not in registered_models, f"Model {test_model_id} should not be pre-registered for this test"

    # Check if anthropic provider is available in ci-tests
    providers = {p.provider_id: p for p in client.providers.list()}
    if "anthropic" not in providers:
        pytest.skip("Anthropic provider not configured in ci-tests - cannot test unregistered model routing")

    # Get the actual provider implementation from the library client's stack
    inference_router = client.async_client.impls.get(Api.inference)
    if not inference_router:
        raise RuntimeError("No inference router found")

    # The inference router's routing_table.impls_by_provider_id should have anthropic
    # Let's patch the anthropic provider's openai_chat_completion method
    # to avoid making real API calls
    mock_response = OpenAIChatCompletion(
        id="chatcmpl-test-123",
        created=1234567890,
        model="claude-3-5-sonnet-20241022",
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason="stop",
                message=OpenAIChatCompletionResponseMessage(
                    content="Mocked response to test routing",
                ),
            )
        ],
        usage=OpenAIChatCompletionUsage(
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
        ),
    )

    # Get the routing table from the inference router
    routing_table = inference_router.routing_table

    # Patch the anthropic provider's openai_chat_completion method
    anthropic_provider = routing_table.impls_by_provider_id.get("anthropic")
    if not anthropic_provider:
        raise RuntimeError("Anthropic provider not found in routing table even though it's in providers list")

    with patch.object(
        anthropic_provider,
        "openai_chat_completion",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_method:
        # Make the request with the unregistered model
        response = client.chat.completions.create(
            model=test_model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Test message for unregistered model routing",
                }
            ],
            stream=False,
        )

        # Verify the provider's method was called
        assert mock_method.called, "Provider's openai_chat_completion should have been called"

        # Verify the response came through
        assert response.choices[0].message.content == "Mocked response to test routing"

        # Verify that the router passed the correct model to the provider
        # (without the "anthropic/" prefix)
        call_args = mock_method.call_args
        params = call_args[0][0]  # First positional argument is the params object
        assert params.model == "claude-3-5-sonnet-20241022", (
            f"Provider should receive model without provider prefix, got {params.model}"
        )
