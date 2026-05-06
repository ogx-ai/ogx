# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for conversation service lifecycle, compatibility, and validation.

0. Purpose: validate conversation service behavior and OpenAI compatibility.
1. Categories: lifecycle CRUD, validation errors, provider compatibility, regression.
2. Tests: lifecycle create/read/delete; item add/list/retrieve; ID validation; empty params; OpenAI adapters; deprecated fields; policy config; regression for missing message type.
"""

import tempfile
from pathlib import Path

import pytest
from openai.types.conversations.conversation import Conversation as OpenAIConversation
from openai.types.conversations.conversation_item import ConversationItem as OpenAIConversationItem
from pydantic import TypeAdapter

from ogx.core.conversations.conversations import (
    ConversationServiceConfig,
    ConversationServiceImpl,
)
from ogx.core.datatypes import StackConfig
from ogx.core.storage.datatypes import (
    ServerStoresConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx_api import (
    ConversationItemNotFoundError,
    ConversationNotFoundError,
    InvalidParameterError,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
)
from ogx_api.conversations import (
    AddItemsRequest,
    CreateConversationRequest,
    DeleteConversationRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
)


@pytest.fixture
async def service():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conversations.db"

        storage = StorageConfig(
            backends={
                "sql_test": SqliteSqlStoreConfig(db_path=str(db_path)),
            },
            stores=ServerStoresConfig(
                conversations=SqlStoreReference(backend="sql_test", table_name="openai_conversations"),
                metadata=None,
                inference=None,
                prompts=None,
            ),
        )
        register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})
        stack_config = StackConfig(distro_name="test", apis=[], providers={}, storage=storage)

        config = ConversationServiceConfig(config=stack_config, policy=[])
        service = ConversationServiceImpl(config, {})
        await service.initialize()
        yield service


async def test_conversation_lifecycle(service):
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "data"}))

    assert conversation.id.startswith("conv_")
    assert conversation.metadata == {"test": "data"}

    retrieved = await service.get_conversation(GetConversationRequest(conversation_id=conversation.id))
    assert retrieved.id == conversation.id

    deleted = await service.openai_delete_conversation(DeleteConversationRequest(conversation_id=conversation.id))
    assert deleted.id == conversation.id


async def test_conversation_items(service):
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_test123",
            status="completed",
        )
    ]
    item_list = await service.add_items(conversation.id, AddItemsRequest(items=items))

    assert len(item_list.data) == 1
    assert item_list.data[0].id == "msg_test123"

    items_result = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    assert len(items_result.data) == 1


async def test_invalid_conversation_id(service):
    with pytest.raises(InvalidParameterError, match="Conversation ID must match format"):
        await service.get_conversation(GetConversationRequest(conversation_id="invalid_id"))


async def test_invalid_conversation_id_on_retrieve(service):
    with pytest.raises(InvalidParameterError, match="Conversation ID must match format"):
        await service.retrieve(RetrieveItemRequest(conversation_id="bad_id", item_id="item_123"))


async def test_invalid_conversation_id_on_update(service):
    from ogx_api.conversations import UpdateConversationRequest

    with pytest.raises(InvalidParameterError, match="Conversation ID must match format"):
        await service.update_conversation("bad_id", UpdateConversationRequest(metadata={}))


async def test_invalid_conversation_id_on_delete(service):
    with pytest.raises(InvalidParameterError, match="Conversation ID must match format"):
        await service.openai_delete_conversation(DeleteConversationRequest(conversation_id="bad_id"))


async def test_nonexistent_conversation_raises_conversation_not_found(service):
    """Test that get_conversation raises ConversationNotFoundError for nonexistent ID."""
    nonexistent_id = "conv_" + "0" * 48
    with pytest.raises(ConversationNotFoundError, match=f"Conversation '{nonexistent_id}' not found"):
        await service.get_conversation(GetConversationRequest(conversation_id=nonexistent_id))


async def test_retrieve_nonexistent_item_raises_conversation_item_not_found(service):
    """Test that retrieve raises ConversationItemNotFoundError for nonexistent item."""
    conversation = await service.create_conversation(CreateConversationRequest())
    with pytest.raises(
        ConversationItemNotFoundError,
        match="Conversation item 'msg_nonexistent' not found in conversation",
    ):
        await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id="msg_nonexistent"))


async def test_openai_type_compatibility(service):
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "value"}))

    conversation_dict = conversation.model_dump()
    openai_conversation = OpenAIConversation.model_validate(conversation_dict)

    for attr in ["id", "object", "created_at", "metadata"]:
        assert getattr(openai_conversation, attr) == getattr(conversation, attr)

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_test456",
            status="completed",
        )
    ]
    item_list = await service.add_items(conversation.id, AddItemsRequest(items=items))

    for attr in ["object", "data", "first_id", "last_id", "has_more"]:
        assert hasattr(item_list, attr)
    assert item_list.object == "list"

    items_result = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    item = await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id=items_result.data[0].id))
    item_dict = item.model_dump()

    openai_item_adapter = TypeAdapter(OpenAIConversationItem)
    openai_item_adapter.validate_python(item_dict)


async def test_items_not_returned_on_creation_or_retrieval(service):
    """Test that items field is not returned when creating or retrieving a conversation.

    The items field is deprecated and kept for backward compatibility.
    Items should be accessed via the conversation_items table using the /items endpoint.
    """
    # Create a conversation
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "value"}))

    # Verify items field doesn't exist in serialized response
    conversation_dict = conversation.model_dump(exclude_none=True)
    assert "items" not in conversation_dict, "items should not be in creation response"

    # Retrieve the conversation
    retrieved = await service.get_conversation(GetConversationRequest(conversation_id=conversation.id))

    # Verify items field doesn't exist in retrieval response
    retrieved_dict = retrieved.model_dump(exclude_none=True)
    assert "items" not in retrieved_dict, "items should not be in retrieval response"


async def test_policy_configuration():
    from ogx.core.access_control.datatypes import Action, Scope
    from ogx.core.datatypes import AccessRule

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conversations_policy.db"

        restrictive_policy = [
            AccessRule(forbid=Scope(principal="test_user", actions=[Action.CREATE, Action.READ], resource="*"))
        ]

        storage = StorageConfig(
            backends={
                "sql_test": SqliteSqlStoreConfig(db_path=str(db_path)),
            },
            stores=ServerStoresConfig(
                conversations=SqlStoreReference(backend="sql_test", table_name="openai_conversations"),
                metadata=None,
                inference=None,
                prompts=None,
            ),
        )
        register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})
        stack_config = StackConfig(distro_name="test", apis=[], providers={}, storage=storage)

        config = ConversationServiceConfig(config=stack_config, policy=restrictive_policy)
        service = ConversationServiceImpl(config, {})
        await service.initialize()

        assert service.policy == restrictive_policy
        assert len(service.policy) == 1
        assert service.policy[0].forbid is not None


async def test_add_items_defaults_message_type(service):
    items = [
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]

    conversation = await service.create_conversation(CreateConversationRequest())

    added = await service.add_items(conversation.id, AddItemsRequest(items=items))

    assert len(added.data) == 1
    assert added.data[0].type == "message"


async def test_create_conversation_defaults_message_type(service):
    items = [
        {"role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]},
    ]

    conversation = await service.create_conversation(CreateConversationRequest(items=items))

    listed = await service.list_items(ListItemsRequest(conversation_id=conversation.id))

    assert len(listed.data) == 1
    assert listed.data[0].type == "message"
