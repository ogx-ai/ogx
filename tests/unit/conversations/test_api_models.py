# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import Conversation, ConversationItem, ConversationItemList


def test_conversation_model_defaults():
    conversation = Conversation(
        id="conv_123456789",
        created_at=1234567890,
        metadata=None,
        object="conversation",
    )
    assert conversation.id == "conv_123456789"
    assert conversation.object == "conversation"
    assert conversation.metadata is None


def test_openai_client_compatibility():
    from openai.types.conversations.message import Message
    from pydantic import TypeAdapter

    openai_message = Message(
        id="msg_123",
        content=[{"type": "input_text", "text": "Hello"}],
        role="user",
        status="in_progress",
        type="message",
        object="message",
    )

    adapter = TypeAdapter(ConversationItem)
    validated_item = adapter.validate_python(openai_message.model_dump())

    assert validated_item.id == "msg_123"
    assert validated_item.type == "message"


def test_conversation_item_list():
    item_list = ConversationItemList(data=[])
    assert item_list.object == "list"
    assert item_list.data == []
    assert item_list.first_id is None
    assert item_list.last_id is None
    assert item_list.has_more is False
