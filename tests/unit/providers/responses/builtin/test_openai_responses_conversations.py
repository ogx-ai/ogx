# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage

# Fixtures imported from test_openai_responses via root conftest.py for pytest 8.4+ compatibility
from ogx.providers.inline.responses.builtin.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from ogx_api.common.errors import (
    ConversationNotFoundError,
    InvalidParameterError,
)
from ogx_api.conversations import (
    ConversationItemList,
)
from ogx_api.openai_responses import (
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseOutputMessageContentOutputText,
)


@pytest.fixture
def responses_impl_with_conversations(
    mock_inference_api,
    mock_tool_groups_api,
    mock_tool_runtime_api,
    mock_responses_store,
    mock_vector_io_api,
    mock_conversations_api,
    mock_safety_api,
    mock_prompts_api,
    mock_files_api,
    mock_connectors_api,
):
    """Create OpenAIResponsesImpl instance with conversations API."""
    return OpenAIResponsesImpl(
        inference_api=mock_inference_api,
        tool_groups_api=mock_tool_groups_api,
        tool_runtime_api=mock_tool_runtime_api,
        responses_store=mock_responses_store,
        vector_io_api=mock_vector_io_api,
        conversations_api=mock_conversations_api,
        safety_api=mock_safety_api,
        prompts_api=mock_prompts_api,
        files_api=mock_files_api,
        connectors_api=mock_connectors_api,
    )


class TestConversationValidation:
    """Test conversation ID validation logic."""

    async def test_nonexistent_conversation_raises_error(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test that ConversationNotFoundError is raised for non-existent conversation."""
        conv_id = "conv_" + "0" * 48

        # Mock conversation not found
        mock_conversations_api.list_items.side_effect = ConversationNotFoundError(conv_id)

        with pytest.raises(ConversationNotFoundError):
            await responses_impl_with_conversations.create_openai_response(
                input="Hello", model="test-model", conversation=conv_id, stream=False
            )


class TestMessageSyncing:
    """Test message syncing to conversations."""

    async def test_sync_response_to_conversation_simple(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test syncing simple response to conversation."""
        conv_id = "conv_test123"
        input_text = "What are the 5 Ds of dodgeball?"

        # Output items (what the model generated)
        output_items = [
            OpenAIResponseMessage(
                id="msg_response",
                content=[
                    OpenAIResponseOutputMessageContentOutputText(
                        text="The 5 Ds are: Dodge, Duck, Dip, Dive, and Dodge.", type="output_text", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        await responses_impl_with_conversations._sync_response_to_conversation(conv_id, input_text, output_items)

        # should call add_items with user input and assistant response
        mock_conversations_api.add_items.assert_called_once()
        call_args = mock_conversations_api.add_items.call_args

        assert call_args[0][0] == conv_id  # conversation_id
        request = call_args[0][1]  # AddItemsRequest
        items = request.items

        assert len(items) == 2
        # User message
        assert items[0].type == "message"
        assert items[0].role == "user"
        assert items[0].content[0].type == "input_text"
        assert items[0].content[0].text == input_text

        # Assistant message
        assert items[1].type == "message"
        assert items[1].role == "assistant"

    async def test_sync_response_to_conversation_api_error(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        mock_conversations_api.add_items.side_effect = Exception("API Error")
        output_items = []

        # matching the behavior of OpenAI here
        with pytest.raises(Exception, match="API Error"):
            await responses_impl_with_conversations._sync_response_to_conversation(
                "conv_test123", "Hello", output_items
            )

    async def test_sync_with_list_input(self, responses_impl_with_conversations, mock_conversations_api):
        """Test syncing with list of input messages."""
        conv_id = "conv_test123"
        input_messages = [
            OpenAIResponseMessage(role="user", content=[{"type": "input_text", "text": "First message"}]),
        ]
        output_items = [
            OpenAIResponseMessage(
                id="msg_response",
                content=[OpenAIResponseOutputMessageContentOutputText(text="Response", type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ]

        await responses_impl_with_conversations._sync_response_to_conversation(conv_id, input_messages, output_items)

        mock_conversations_api.add_items.assert_called_once()
        call_args = mock_conversations_api.add_items.call_args

        request = call_args[0][1]  # AddItemsRequest
        items = request.items
        # Should have input message + output message
        assert len(items) == 2


class TestIntegrationWorkflow:
    """Integration tests for the full conversation workflow."""

    async def test_create_response_with_valid_conversation(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test creating a response with a valid conversation parameter."""
        mock_conversations_api.list_items.return_value = ConversationItemList(
            data=[], first_id=None, has_more=False, last_id=None, object="list"
        )

        async def mock_streaming_response(*args, **kwargs):
            message_item = OpenAIResponseMessage(
                id="msg_response",
                content=[
                    OpenAIResponseOutputMessageContentOutputText(
                        text="Test response", type="output_text", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )

            # Emit output_item.done event first (needed for conversation sync)
            yield OpenAIResponseObjectStreamResponseOutputItemDone(
                response_id="resp_test123",
                item=message_item,
                output_index=0,
                sequence_number=1,
                type="response.output_item.done",
            )

            # Then emit response.completed
            mock_response = OpenAIResponseObject(
                id="resp_test123",
                created_at=1234567890,
                model="test-model",
                object="response",
                output=[message_item],
                status="completed",
                store=True,
            )

            yield OpenAIResponseObjectStreamResponseCompleted(
                response=mock_response, sequence_number=2, type="response.completed"
            )

        responses_impl_with_conversations._create_streaming_response = mock_streaming_response

        input_text = "Hello, how are you?"
        conversation_id = "conv_" + "a" * 48

        response = await responses_impl_with_conversations.create_openai_response(
            input=input_text, model="test-model", conversation=conversation_id, stream=False
        )

        assert response is not None
        assert response.id == "resp_test123"

        # Note: conversation sync happens inside _create_streaming_response,
        # which we're mocking here, so we can't test it in this unit test.
        # The sync logic is tested separately in TestMessageSyncing.

    async def test_create_response_with_invalid_conversation_id(self, responses_impl_with_conversations):
        """Test creating a response with an invalid conversation ID."""
        with pytest.raises(
            InvalidParameterError, match="Must match format 'conv_' followed by 48 lowercase hex characters"
        ):
            await responses_impl_with_conversations.create_openai_response(
                input="Hello", model="test-model", conversation="invalid_id", stream=False
            )

    async def test_create_response_with_nonexistent_conversation(
        self, responses_impl_with_conversations, mock_conversations_api
    ):
        """Test creating a response with a non-existent conversation."""
        conv_id = "conv_" + "b" * 48
        mock_conversations_api.list_items.side_effect = ConversationNotFoundError(conv_id)

        with pytest.raises(ConversationNotFoundError) as exc_info:
            await responses_impl_with_conversations.create_openai_response(
                input="Hello", model="test-model", conversation=conv_id, stream=False
            )

        assert "not found" in str(exc_info.value)

    async def test_conversation_and_previous_response_id(
        self, responses_impl_with_conversations, mock_conversations_api, mock_responses_store
    ):
        with pytest.raises(InvalidParameterError, match="Provide only one") as exc_info:
            await responses_impl_with_conversations.create_openai_response(
                input="test", model="test", conversation="conv_123", previous_response_id="resp_123"
            )

        assert "previous_response_id" in str(exc_info.value)
        assert "conversation" in str(exc_info.value)


class TestStoreFalseConversationLeak:
    """Regression tests for https://github.com/ogx-ai/ogx/issues/5304

    When store=False, conversation messages must NOT be synced to the database.
    """

    @staticmethod
    async def _fake_stream():
        yield ChatCompletionChunk(
            id="chatcmpl-store-test",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="4", role="assistant"),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )

    async def test_store_false_does_not_sync_conversation(
        self,
        responses_impl_with_conversations,
        mock_inference_api,
        mock_responses_store,
        mock_conversations_api,
    ):
        """store=False with a conversation ID must not write to the conversation store."""
        conv_id = "conv_" + "a" * 48
        mock_conversations_api.list_items.return_value = ConversationItemList(
            data=[], first_id=None, has_more=False, last_id=None, object="list"
        )
        mock_responses_store.get_conversation_messages.return_value = None
        mock_inference_api.openai_chat_completion.return_value = self._fake_stream()

        result = await responses_impl_with_conversations.create_openai_response(
            input="What is 2+2?",
            model="test-model",
            store=False,
            conversation=conv_id,
            stream=False,
        )

        assert result.status == "completed"
        mock_responses_store.store_conversation_messages.assert_not_called()
        mock_conversations_api.add_items.assert_not_called()
        mock_responses_store.upsert_response_object.assert_not_called()

    async def test_store_true_does_sync_conversation(
        self,
        responses_impl_with_conversations,
        mock_inference_api,
        mock_responses_store,
        mock_conversations_api,
    ):
        """store=True with a conversation ID must write to the conversation store."""
        conv_id = "conv_" + "b" * 48
        mock_conversations_api.list_items.return_value = ConversationItemList(
            data=[], first_id=None, has_more=False, last_id=None, object="list"
        )
        mock_responses_store.get_conversation_messages.return_value = None
        mock_inference_api.openai_chat_completion.return_value = self._fake_stream()

        result = await responses_impl_with_conversations.create_openai_response(
            input="What is 2+2?",
            model="test-model",
            store=True,
            conversation=conv_id,
            stream=False,
        )

        assert result.status == "completed"
        mock_responses_store.store_conversation_messages.assert_called_once()
        mock_conversations_api.add_items.assert_called_once()
        mock_responses_store.upsert_response_object.assert_called()

    async def test_store_false_streaming_does_not_sync_conversation(
        self,
        responses_impl_with_conversations,
        mock_inference_api,
        mock_responses_store,
        mock_conversations_api,
    ):
        """store=False in streaming mode must also not write to the conversation store."""
        conv_id = "conv_" + "c" * 48
        mock_conversations_api.list_items.return_value = ConversationItemList(
            data=[], first_id=None, has_more=False, last_id=None, object="list"
        )
        mock_responses_store.get_conversation_messages.return_value = None
        mock_inference_api.openai_chat_completion.return_value = self._fake_stream()

        chunks = []
        async for chunk in await responses_impl_with_conversations.create_openai_response(
            input="What is 2+2?",
            model="test-model",
            store=False,
            conversation=conv_id,
            stream=True,
        ):
            chunks.append(chunk)

        assert any(c.type == "response.completed" for c in chunks)
        mock_responses_store.store_conversation_messages.assert_not_called()
        mock_conversations_api.add_items.assert_not_called()
