# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest


@pytest.mark.integration
class TestConversationResponses:
    """Integration tests for the conversation parameter in responses API."""

    def test_conversation_basic_workflow(self, openai_client, text_model_id):
        """Test basic conversation workflow: create conversation, add response, verify sync."""
        conversation = openai_client.conversations.create(metadata={"topic": "test"})
        assert conversation.id.startswith("conv_")

        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What are the 5 Ds of dodgeball?"}],
            conversation=conversation.id,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0

        # Verify conversation was synced properly
        conversation_items = openai_client.conversations.items.list(conversation.id)
        assert len(conversation_items.data) >= 2

        roles = [item.role for item in conversation_items.data if hasattr(item, "role")]
        assert "user" in roles and "assistant" in roles

    def test_conversation_multi_turn_and_streaming(self, openai_client, text_model_id):
        """Test multi-turn conversations and streaming responses."""
        conversation = openai_client.conversations.create()

        # First turn
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "Say hello"}],
            conversation=conversation.id,
        )

        # Second turn with streaming - use context manager to ensure proper connection cleanup
        with openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "Say goodbye"}],
            conversation=conversation.id,
            stream=True,
        ) as response_stream:
            final_response = None
            for chunk in response_stream:
                if chunk.type == "response.completed":
                    final_response = chunk.response
                    break

        assert response1.id != final_response.id
        assert len(response1.output_text.strip()) > 0
        assert len(final_response.output_text.strip()) > 0

        # Verify all turns are in conversation
        conversation_items = openai_client.conversations.items.list(conversation.id)
        assert len(conversation_items.data) >= 4  # 2 user + 2 assistant messages

    @pytest.mark.timeout(60, method="thread")
    def test_conversation_context_loading(self, openai_client, text_model_id):
        """Test that conversation context is properly loaded for responses."""
        conversation = openai_client.conversations.create(
            items=[
                {"type": "message", "role": "user", "content": "My name is Alice. I like to eat apples."},
                {"type": "message", "role": "assistant", "content": "Hello Alice!"},
            ]
        )

        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What do I like to eat?"}],
            conversation=conversation.id,
        )

        assert "apple" in response.output_text.lower()

    @pytest.mark.timeout(60, method="thread")
    def test_conversation_error_handling(self, openai_client, text_model_id):
        """Test error handling for invalid and nonexistent conversations."""
        # Invalid conversation ID format
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                conversation="invalid_id",
            )
        assert any(word in str(exc_info.value).lower() for word in ["conv", "invalid", "bad"])

        # Nonexistent conversation ID (must be valid format to reach the DB lookup)
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                conversation="conv_" + "0" * 48,
            )
        assert any(word in str(exc_info.value).lower() for word in ["not found", "404"])

    #
    #     response = openai_client.responses.create(
    #         model=text_model_id, input=[{"role": "user", "content": "First response"}]
    #     )
    #     with pytest.raises(Exception) as exc_info:
    #         openai_client.responses.create(
    #             model=text_model_id,
    #             input=[{"role": "user", "content": "Hello"}],
    #             conversation="conv_test123",
    #             previous_response_id=response.id,
    #         )
    #     assert "mutually exclusive" in str(exc_info.value).lower()

    def test_conversation_backward_compatibility(self, openai_client, text_model_id):
        """Test that responses work without conversation parameter (backward compatibility)."""
        response = openai_client.responses.create(
            model=text_model_id, input=[{"role": "user", "content": "Hello world"}]
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0

    # this is not ready yet
    # def test_conversation_compat_client(self, responses_client, text_model_id):
    #     """Test conversation parameter works with compatibility client."""
    #     if not hasattr(responses_client, "conversations"):
    #         pytest.skip("responses_client does not support conversations API")
    #
    #     conversation = responses_client.conversations.create()
    #     response = responses_client.responses.create(
    #         model=text_model_id, input="Tell me a joke", conversation=conversation.id
    #     )
    #
    #     assert response is not None
    #     assert len(response.output_text.strip()) > 0
    #
    #     conversation_items = responses_client.conversations.items.list(conversation.id)
    #     assert len(conversation_items.data) >= 2
