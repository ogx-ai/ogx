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

import sqlite3
import tempfile
from pathlib import Path

import pytest
from openai.types.conversations.conversation import Conversation as OpenAIConversation
from openai.types.conversations.conversation_item import ConversationItem as OpenAIConversationItem
from pydantic import TypeAdapter
from sqlalchemy import event

from ogx.core.conversations.conversations import (
    ITEMS_TABLE,
    LEGACY_ITEMS_TABLE,
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
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
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
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType


def _message(text: str, item_id: str | None = None) -> OpenAIResponseMessage:
    return OpenAIResponseMessage(
        type="message",
        role="user",
        content=[OpenAIResponseInputMessageContentText(type="input_text", text=text)],
        id=item_id,
        status="completed",
    )


def _texts(items: ConversationItemList) -> list[tuple[str, str]]:
    return [(item.id, item.content[0].text) for item in items.data]


async def _make_service(db_path: Path) -> ConversationServiceImpl:
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
    stack_config = StackConfig(distro_name="test", providers={}, storage=storage)

    config = ConversationServiceConfig(config=stack_config, policy=[])
    service = ConversationServiceImpl(config, {})
    await service.initialize()
    return service


@pytest.fixture
async def service():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield await _make_service(Path(tmpdir) / "test_conversations.db")


async def test_conversation_lifecycle(service):
    conversation = await service.create_conversation(CreateConversationRequest(metadata={"test": "data"}))

    assert conversation.id.startswith("conv_")
    assert conversation.metadata == {"test": "data"}

    retrieved = await service.get_conversation(GetConversationRequest(conversation_id=conversation.id))
    assert retrieved.id == conversation.id

    deleted = await service.openai_delete_conversation(DeleteConversationRequest(conversation_id=conversation.id))
    assert deleted.id == conversation.id


async def test_delete_conversation_cascades_to_items(service):
    """Deleting a conversation must remove its items and block further item access."""
    conversation = await service.create_conversation(CreateConversationRequest())
    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_deletecascade123",
            status="completed",
        )
    ]
    await service.add_items(conversation.id, AddItemsRequest(items=items))

    listed = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    assert len(listed.data) == 1
    item_id = listed.data[0].id

    await service.openai_delete_conversation(DeleteConversationRequest(conversation_id=conversation.id))

    with pytest.raises(ConversationNotFoundError):
        await service.list_items(ListItemsRequest(conversation_id=conversation.id))

    with pytest.raises(ConversationNotFoundError):
        await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id=item_id))

    raw_items = await service.sql_store.fetch_all(table=ITEMS_TABLE, where={"conversation_id": conversation.id})
    assert raw_items.data == []


async def test_delete_conversation_is_atomic_when_parent_delete_fails(service):
    """A failed parent delete must roll back the child-row delete in the same operation."""
    conversation = await service.create_conversation(CreateConversationRequest())
    items = [
        OpenAIResponseMessage(
            type="message",
            role="user",
            content=[OpenAIResponseInputMessageContentText(type="input_text", text="Hello")],
            id="msg_atomicdelete123",
            status="completed",
        )
    ]
    added = await service.add_items(conversation.id, AddItemsRequest(items=items))

    listed_before_delete = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    assert len(listed_before_delete.data) == 1

    sql_store_impl = service.sql_store.sql_store
    await sql_store_impl._ensure_engine()
    assert sql_store_impl._engine is not None

    failure_triggered = {"value": False}

    def fail_parent_delete(
        conn, cursor, statement, parameters, context, executemany
    ):  # pragma: no cover - SQLAlchemy callback signature
        if statement.startswith("DELETE FROM openai_conversations"):
            failure_triggered["value"] = True
            raise RuntimeError("Injected parent delete failure")

    event.listen(sql_store_impl._engine.sync_engine, "before_cursor_execute", fail_parent_delete)
    try:
        with pytest.raises(RuntimeError, match="Injected parent delete failure"):
            await service.openai_delete_conversation(DeleteConversationRequest(conversation_id=conversation.id))
    finally:
        event.remove(sql_store_impl._engine.sync_engine, "before_cursor_execute", fail_parent_delete)

    assert failure_triggered["value"] is True

    conversation_after_failure = await service.get_conversation(GetConversationRequest(conversation_id=conversation.id))
    assert conversation_after_failure.id == conversation.id

    listed_after_failure = await service.list_items(ListItemsRequest(conversation_id=conversation.id))
    assert len(listed_after_failure.data) == 1
    assert listed_after_failure.data[0].id == added.data[0].id


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
    assert item_list.data[0].id.startswith("msg_")

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
        stack_config = StackConfig(distro_name="test", providers={}, storage=storage)

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
    added = await service.add_items(conversation.id, AddItemsRequest(items=items))
    item_id = added.data[0].id

    result = await service.openai_delete_conversation_item(
        DeleteItemRequest(conversation_id=conversation.id, item_id=item_id)
    )

    assert isinstance(result, Conversation)
    assert result.id == conversation.id
    assert result.object == "conversation"
    assert result.created_at == conversation.created_at

    # Verify the item was actually deleted
    with pytest.raises(ConversationItemNotFoundError):
        await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id=item_id))


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


async def test_add_item_without_id_gets_server_minted_id(service):
    """Case 1: an item posted without an id is stored under a server-minted id."""
    conversation = await service.create_conversation(CreateConversationRequest())

    added = await service.add_items(conversation.id, AddItemsRequest(items=[_message("hello")]))

    item_id = added.data[0].id
    assert item_id.startswith("msg_")
    assert len(item_id) == len("msg_") + 48
    retrieved = await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id=item_id))
    assert retrieved.content[0].text == "hello"


async def test_add_items_copies_item_from_other_conversation(service):
    """Case 2: an id from conversation A posted to B copies A's item with its original content."""
    conv_a = await service.create_conversation(CreateConversationRequest())
    conv_b = await service.create_conversation(CreateConversationRequest())
    item_id = (await service.add_items(conv_a.id, AddItemsRequest(items=[_message("hello from A")]))).data[0].id

    added = await service.add_items(conv_b.id, AddItemsRequest(items=[_message("content sent for B", item_id)]))

    assert _texts(added) == [(item_id, "hello from A")]
    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conv_b.id))) == [(item_id, "hello from A")]
    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conv_a.id))) == [(item_id, "hello from A")]


async def test_add_items_rejects_id_already_in_conversation(service):
    """Case 3: re-adding an item's id to its own conversation is a 400 and the batch writes nothing."""
    conversation = await service.create_conversation(CreateConversationRequest())
    item_id = (await service.add_items(conversation.id, AddItemsRequest(items=[_message("first")]))).data[0].id

    with pytest.raises(InvalidParameterError, match="Invalid value for 'items'.*Item already in conversation"):
        await service.add_items(
            conversation.id, AddItemsRequest(items=[_message("new item"), _message("second", item_id)])
        )

    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conversation.id))) == [(item_id, "first")]


async def test_create_conversation_copies_item_from_other_conversation(service):
    """Case 4: creating a conversation with an item carrying another conversation's id copies it."""
    conv_a = await service.create_conversation(CreateConversationRequest())
    item_id = (await service.add_items(conv_a.id, AddItemsRequest(items=[_message("hello from A")]))).data[0].id

    conv_b = await service.create_conversation(CreateConversationRequest(items=[_message("ignored", item_id)]))

    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conv_b.id))) == [(item_id, "hello from A")]
    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conv_a.id))) == [(item_id, "hello from A")]


async def test_create_conversation_with_rejected_items_writes_nothing(service):
    """A rejected item batch on create must not leave a conversation row behind."""
    conv_a = await service.create_conversation(CreateConversationRequest())
    item_id = (await service.add_items(conv_a.id, AddItemsRequest(items=[_message("hello from A")]))).data[0].id
    conversations_before = len((await service.sql_store.fetch_all(table="openai_conversations")).data)

    with pytest.raises(InvalidParameterError, match="Item already in conversation"):
        await service.create_conversation(
            CreateConversationRequest(items=[_message("copy", item_id), _message("copy again", item_id)])
        )

    conversations_after = len((await service.sql_store.fetch_all(table="openai_conversations")).data)
    assert conversations_after == conversations_before


async def test_add_items_ignores_unknown_client_ids(service):
    """Case 5: two new messages sharing an unknown client id are both stored under fresh server ids."""
    conversation = await service.create_conversation(CreateConversationRequest())
    client_id = "msg_" + "f" * 48

    added = await service.add_items(
        conversation.id, AddItemsRequest(items=[_message("one", client_id), _message("two", client_id)])
    )

    ids = [item.id for item in added.data]
    assert len(set(ids)) == 2
    assert client_id not in ids
    assert all(item_id.startswith("msg_") for item_id in ids)
    listed = await service.list_items(ListItemsRequest(conversation_id=conversation.id, order="asc"))
    assert _texts(listed) == [(ids[0], "one"), (ids[1], "two")]
    with pytest.raises(ConversationItemNotFoundError):
        await service.retrieve(RetrieveItemRequest(conversation_id=conversation.id, item_id=client_id))


async def test_delete_item_leaves_copy_in_other_conversation(service):
    """Case 6: deleting A's item after it was copied into B leaves B's copy in place."""
    conv_a = await service.create_conversation(CreateConversationRequest())
    conv_b = await service.create_conversation(CreateConversationRequest())
    item_id = (await service.add_items(conv_a.id, AddItemsRequest(items=[_message("hello from A")]))).data[0].id
    await service.add_items(conv_b.id, AddItemsRequest(items=[_message("ignored", item_id)]))

    await service.openai_delete_conversation_item(DeleteItemRequest(conversation_id=conv_a.id, item_id=item_id))

    with pytest.raises(ConversationItemNotFoundError):
        await service.retrieve(RetrieveItemRequest(conversation_id=conv_a.id, item_id=item_id))
    retrieved = await service.retrieve(RetrieveItemRequest(conversation_id=conv_b.id, item_id=item_id))
    assert retrieved.content[0].text == "hello from A"


async def test_sync_items_keeps_ids_and_skips_items_already_present(service):
    """The Responses sync keeps the ids it is given and does not duplicate or reject re-sent items."""
    conversation = await service.create_conversation(CreateConversationRequest())
    reply_id = "msg_" + "1" * 48
    next_id = "msg_" + "2" * 48

    await service.sync_items(conversation.id, AddItemsRequest(items=[_message("hello"), _message("reply", reply_id)]))
    first_turn = await service.list_items(ListItemsRequest(conversation_id=conversation.id, order="asc"))
    assert first_turn.data[0].id.startswith("msg_")
    assert _texts(first_turn)[1:] == [(reply_id, "reply")]

    await service.sync_items(
        conversation.id, AddItemsRequest(items=[_message("edited reply", reply_id), _message("next", next_id)])
    )

    second_turn = await service.list_items(ListItemsRequest(conversation_id=conversation.id, order="asc"))
    assert _texts(second_turn)[1:] == [(reply_id, "reply"), (next_id, "next")]


async def test_sync_items_allows_id_present_in_other_conversation(service):
    """Output items replayed into another conversation keep their id there as well."""
    conv_a = await service.create_conversation(CreateConversationRequest())
    conv_b = await service.create_conversation(CreateConversationRequest())
    item_id = "msg_" + "3" * 48

    await service.sync_items(conv_a.id, AddItemsRequest(items=[_message("from A", item_id)]))
    await service.sync_items(conv_b.id, AddItemsRequest(items=[_message("from B", item_id)]))

    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conv_a.id))) == [(item_id, "from A")]
    assert _texts(await service.list_items(ListItemsRequest(conversation_id=conv_b.id))) == [(item_id, "from B")]


async def test_list_items_after_cursor_uses_position_in_this_conversation(service):
    """An id copied into several conversations must page from its position in the listed one."""
    conv_a = await service.create_conversation(CreateConversationRequest())
    conv_b = await service.create_conversation(CreateConversationRequest())
    shared_id = (await service.add_items(conv_a.id, AddItemsRequest(items=[_message("shared")]))).data[0].id
    await service.add_items(conv_b.id, AddItemsRequest(items=[_message("b0"), _message("b1")]))
    await service.add_items(conv_b.id, AddItemsRequest(items=[_message("ignored", shared_id)]))
    await service.add_items(conv_b.id, AddItemsRequest(items=[_message("b3")]))

    page = await service.list_items(ListItemsRequest(conversation_id=conv_b.id, order="asc", after=shared_id))

    assert [item.content[0].text for item in page.data] == ["b3"]


async def test_initialize_copies_legacy_conversation_items():
    """Rows written to the id-keyed table by earlier servers are readable after upgrade, once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "legacy.db"
        conversation_id = "conv_" + "a" * 48
        legacy_rows = [
            {
                "id": f"msg_{i}",
                "conversation_id": conversation_id,
                "created_at": 1700000000,
                "sort_order": i,
                "item_data": _message(f"legacy {i}", f"msg_{i}").model_dump(),
                "owner_principal": "",
                "access_attributes": None,
            }
            for i in range(2)
        ]
        legacy_store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=str(db_path)))
        await legacy_store.create_table(
            LEGACY_ITEMS_TABLE,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "conversation_id": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "sort_order": ColumnType.INTEGER,
                "item_data": ColumnType.JSON,
                "owner_principal": ColumnType.STRING,
                "access_attributes": ColumnType.JSON,
            },
        )
        await legacy_store.create_table(
            "openai_conversations",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "items": ColumnType.JSON,
                "metadata": ColumnType.JSON,
                "owner_principal": ColumnType.STRING,
                "access_attributes": ColumnType.JSON,
            },
        )
        await legacy_store.insert(
            "openai_conversations",
            {"id": conversation_id, "created_at": 1700000000, "metadata": None, "owner_principal": ""},
        )
        await legacy_store.insert(LEGACY_ITEMS_TABLE, legacy_rows)
        await legacy_store.shutdown()

        for _ in range(2):
            service = await _make_service(db_path)
            listed = await service.list_items(ListItemsRequest(conversation_id=conversation_id, order="asc"))
            assert _texts(listed) == [("msg_0", "legacy 0"), ("msg_1", "legacy 1")]
            await service.sql_store.sql_store.shutdown()

        service = await _make_service(db_path)
        other = await service.create_conversation(CreateConversationRequest(items=[_message("x", "msg_0")]))
        assert _texts(await service.list_items(ListItemsRequest(conversation_id=other.id))) == [("msg_0", "legacy 0")]
        with sqlite3.connect(db_path) as legacy_db:
            assert legacy_db.execute(f"SELECT COUNT(*) FROM {LEGACY_ITEMS_TABLE}").fetchone() == (2,)


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


async def test_item_ordering_uses_sort_order_not_timestamp(service):
    """sort_order column must exist and increase monotonically across add_items calls.

    Regression: created_at uses second-precision timestamps, so calls within the
    same second collide. Ordering must use a dedicated sort_order counter, not
    created_at. We assert on the column values directly so this fails if
    sort_order is missing or not populated, regardless of DB-specific row ordering.
    """
    from unittest.mock import patch

    conversation = await service.create_conversation(CreateConversationRequest())
    fixed_time = 1700000000

    with patch("ogx.core.conversations.conversations.time") as mock_time:
        mock_time.time.return_value = fixed_time

        for i in range(5):
            items = [
                OpenAIResponseMessage(
                    type="message",
                    role="user",
                    content=[OpenAIResponseInputMessageContentText(type="input_text", text=f"msg-{i}")],
                    id=f"msg_{'0' * 44}{i:04d}",
                    status="completed",
                )
            ]
            await service.add_items(conversation.id, AddItemsRequest(items=items))

    raw = await service.sql_store.fetch_all(
        table=ITEMS_TABLE,
        where={"conversation_id": conversation.id},
        order_by=[("sort_order", "asc")],
    )

    sort_orders = [r["sort_order"] for r in raw.data]
    assert sort_orders == [0, 1, 2, 3, 4]

    timestamps = {r["created_at"] for r in raw.data}
    assert len(timestamps) == 1, f"All timestamps should be identical, got {timestamps}"


async def test_list_items_empty_conversation(service):
    """Listing items on empty conversation returns valid ConversationItemList with empty strings."""
    conversation = await service.create_conversation(CreateConversationRequest())

    result = await service.list_items(ListItemsRequest(conversation_id=conversation.id))

    assert len(result.data) == 0
    assert result.has_more is False
    assert result.first_id == ""
    assert result.last_id == ""
