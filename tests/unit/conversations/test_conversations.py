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
    DeleteItemRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
)
from ogx_api.conversations.models import (
    Conversation,
    ConversationDeletedResource,
    ConversationItemList,
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
                connectors=None,
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
                connectors=None,
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


def test_conversation_model_has_no_items_field():
    """Conversation response model should not have an items field per OpenAI spec."""
    conv = Conversation(id="conv_" + "a" * 48, created_at=1000, metadata=None)
    assert "items" not in conv.model_fields


def test_conversation_deleted_resource_object_literal():
    """ConversationDeletedResource.object must be the literal 'conversation.deleted'."""
    deleted = ConversationDeletedResource(id="conv_" + "a" * 48)
    assert deleted.object == "conversation.deleted"
    schema = ConversationDeletedResource.model_json_schema()
    obj_schema = schema["properties"]["object"]
    assert obj_schema.get("const") == "conversation.deleted" or obj_schema.get("enum") == ["conversation.deleted"]


def test_conversation_item_list_object_literal():
    """ConversationItemList.object must be the literal 'list'."""
    schema = ConversationItemList.model_json_schema()
    obj_schema = schema["properties"]["object"]
    assert obj_schema.get("const") == "list" or obj_schema.get("enum") == ["list"]


def test_conversation_item_list_first_last_id_required():
    """first_id and last_id must be required and nullable strings."""
    schema = ConversationItemList.model_json_schema()
    assert "first_id" in schema.get("required", [])
    assert "last_id" in schema.get("required", [])
    first_id_schema = schema["properties"]["first_id"]
    last_id_schema = schema["properties"]["last_id"]
    for field_schema in (first_id_schema, last_id_schema):
        types = field_schema.get("anyOf", [])
        type_names = {t.get("type") for t in types}
        assert {"string", "null"} == type_names


async def test_delete_item_returns_parent_conversation(service):
    """Deleting an item returns the parent Conversation object per OpenAI spec."""
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_todelete",
            status="completed",
        )
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    result = await service.openai_delete_conversation_item(
        DeleteItemRequest(conversation_id=conversation.id, item_id="msg_todelete")
    )

    assert isinstance(result, Conversation)
    assert result.id == conversation.id
    assert result.object == "conversation"
    assert result.created_at == conversation.created_at

    # Verify the item was actually deleted
    with pytest.raises(ConversationItemNotFoundError):
        await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id="msg_todelete"))


async def test_list_items_has_more_with_limit(service):
    """has_more should be True when more items exist beyond the limit."""
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text=f"Message {i}")],
            id=f"msg_{'0' * 44}{i:04d}",
            status="completed",
        )
        for i in range(5)
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    result = await service.list_items(ListItemsRequest(conversation_id=conversation.id, limit=3))

    assert len(result.data) == 3
    assert result.has_more is True
    assert result.first_id != ""
    assert result.last_id != ""


async def test_list_items_has_more_false_when_all_fit(service):
    """has_more should be False when all items fit within the limit."""
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Only one")],
            id="msg_" + "a" * 48,
            status="completed",
        )
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    result = await service.list_items(ListItemsRequest(conversation_id=conversation.id, limit=20))

    assert len(result.data) == 1
    assert result.has_more is False


async def test_list_items_after_cursor(service):
    """after parameter should return items after the given cursor."""
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text=f"Message {i}")],
            id=f"msg_{'0' * 44}{i:04d}",
            status="completed",
        )
        for i in range(5)
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    # List all items first (desc order = newest first)
    all_items = await service.list_items(ListItemsRequest(conversation_id=conversation.id, limit=100))
    assert len(all_items.data) == 5

    # Use the second item as cursor — should get items after it (older items in desc)
    cursor_id = all_items.data[1].id
    result = await service.list_items(ListItemsRequest(conversation_id=conversation.id, after=cursor_id, limit=100))

    assert len(result.data) == 3
    for item in result.data:
        assert item.id != cursor_id


async def test_list_items_after_cursor_with_asc_order(service):
    """after parameter with asc order should return items after the cursor in ascending order."""
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text=f"Message {i}")],
            id=f"msg_{'0' * 44}{i:04d}",
            status="completed",
        )
        for i in range(5)
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    # List all in asc order (oldest first)
    all_items = await service.list_items(ListItemsRequest(conversation_id=conversation.id, order="asc", limit=100))
    assert len(all_items.data) == 5

    # Use the second item as cursor — should get items after it (newer items in asc)
    cursor_id = all_items.data[1].id
    result = await service.list_items(
        ListItemsRequest(conversation_id=conversation.id, after=cursor_id, order="asc", limit=100)
    )

    assert len(result.data) == 3


async def test_list_items_after_cursor_with_has_more(service):
    """after cursor combined with limit should correctly compute has_more."""
    conversation = await service.create_conversation(CreateConversationRequest())

    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text=f"Message {i}")],
            id=f"msg_{'0' * 44}{i:04d}",
            status="completed",
        )
        for i in range(10)
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    # Get all items in desc order
    all_items = await service.list_items(ListItemsRequest(conversation_id=conversation.id, limit=100))
    assert len(all_items.data) == 10

    # Cursor at item 3 (4th from top in desc), limit=3
    # Items after cursor: 6 remaining, limit 3, so has_more=True
    cursor_id = all_items.data[3].id
    result = await service.list_items(ListItemsRequest(conversation_id=conversation.id, after=cursor_id, limit=3))

    assert len(result.data) == 3
    assert result.has_more is True


async def test_list_items_after_invalid_cursor_raises_error(service):
    """after parameter with nonexistent item ID should raise an error."""
    conversation = await service.create_conversation(CreateConversationRequest())

    with pytest.raises(ConversationItemNotFoundError):
        await service.list_items(ListItemsRequest(conversation_id=conversation.id, after="msg_nonexistent"))


async def test_list_items_after_cursor_from_other_conversation_raises_error(service):
    """after cursor from a different conversation should raise an error, not silently return wrong results."""
    conv1 = await service.create_conversation(CreateConversationRequest())
    conv2 = await service.create_conversation(CreateConversationRequest())

    conv1_items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello from conv1")],
            id="msg_" + "b" * 48,
            status="completed",
        )
    ]
    conv2_items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello from conv2")],
            id="msg_" + "c" * 48,
            status="completed",
        )
    ]
    await service.add_items(conv1.id, AddItemsRequest(items=conv1_items))
    await service.add_items(conv2.id, AddItemsRequest(items=conv2_items))

    # Get the item ID from conv1
    listed = await service.list_items(ListItemsRequest(conversation_id=conv1.id))
    cursor_from_conv1 = listed.data[0].id

    # Using conv1's cursor on conv2 should raise an error
    with pytest.raises(ConversationItemNotFoundError):
        await service.list_items(ListItemsRequest(conversation_id=conv2.id, after=cursor_from_conv1))


async def test_create_conversation_with_items_supports_pagination(service):
    """Items created via create_conversation should have unique timestamps for correct pagination."""
    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text=f"Initial {i}")],
            id=f"msg_{'0' * 44}{i:04d}",
            status="completed",
        )
        for i in range(5)
    ]
    conversation = await service.create_conversation(CreateConversationRequest(items=items))

    # Paginate with limit=2 to verify no items are lost
    all_ids = set()
    after = None
    pages = 0
    while True:
        result = await service.list_items(
            ListItemsRequest(conversation_id=conversation.id, limit=2, order="asc", after=after)
        )
        for item in result.data:
            all_ids.add(item.id)
        pages += 1
        if not result.has_more:
            break
        after = result.last_id

    assert len(all_ids) == 5, f"Expected 5 items across all pages, got {len(all_ids)}"
    assert pages == 3


async def test_list_items_empty_conversation(service):
    """Listing items on empty conversation returns valid ConversationItemList with empty strings."""
    conversation = await service.create_conversation(CreateConversationRequest())

    result = await service.list_items(ListItemsRequest(conversation_id=conversation.id))

    assert len(result.data) == 0
    assert result.has_more is False
    assert result.first_id == ""
    assert result.last_id == ""
