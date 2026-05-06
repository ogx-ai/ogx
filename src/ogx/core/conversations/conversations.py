# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import secrets
import time
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ogx.core.access_control.datatypes import AccessRule
from ogx.core.conversations.validation import CONVERSATION_ID_PATTERN
from ogx.core.datatypes import StackConfig
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx.log import get_logger
from ogx_api import (
    Api,
    ConversationItemNotFoundError,
    ConversationNotFoundError,
    InvalidParameterError,
    ServiceNotEnabledError,
)
from ogx_api.conversations import (
    AddItemsRequest,
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemDeletedResource,
    ConversationItemList,
    Conversations,
    CreateConversationRequest,
    DeleteConversationRequest,
    DeleteItemRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
    UpdateConversationRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="openai_conversations")


class ConversationServiceConfig(BaseModel):
    """Configuration for the built-in conversation service.

    :param run_config: Stack run configuration for resolving persistence
    :param policy: Access control rules
    """

    config: StackConfig
    policy: list[AccessRule] = []


async def get_provider_impl(config: ConversationServiceConfig, deps: dict[Api, Any]) -> "ConversationServiceImpl":
    """Get the conversation service implementation."""
    impl = ConversationServiceImpl(config, deps)
    await impl.initialize()
    return impl


class ConversationServiceImpl(Conversations):
    """Built-in conversation service implementation using AuthorizedSqlStore."""

    def __init__(self, config: ConversationServiceConfig, deps: dict[Api, Any]):
        self.config = config
        self.deps = deps
        self.policy = config.policy

        # Use conversations store reference from run config
        conversations_ref = config.config.storage.stores.conversations
        if not conversations_ref:
            raise ServiceNotEnabledError("storage.stores.conversations")

        base_sql_store = sqlstore_impl(conversations_ref)
        self.sql_store = AuthorizedSqlStore(base_sql_store, self.policy)

    async def initialize(self) -> None:
        """Initialize the store and create tables."""
        await self.sql_store.create_table(
            "openai_conversations",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "items": ColumnType.JSON,  # Deprecated: kept for backward compatibility, use conversation_items table instead
                "metadata": ColumnType.JSON,
            },
        )

        await self.sql_store.create_table(
            "conversation_items",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "conversation_id": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "item_data": ColumnType.JSON,
            },
        )

    async def create_conversation(self, request: CreateConversationRequest) -> Conversation:
        """Create a conversation."""
        random_bytes = secrets.token_bytes(24)
        conversation_id = f"conv_{random_bytes.hex()}"
        created_at = int(time.time())

        record_data = {
            "id": conversation_id,
            "created_at": created_at,
            "metadata": request.metadata,
        }

        await self.sql_store.insert(
            table="openai_conversations",
            data=record_data,
        )

        if request.items:
            item_records = []
            for item in request.items:
                item_dict = item.model_dump()
                item_id = self._get_or_generate_item_id(item, item_dict)

                item_record = {
                    "id": item_id,
                    "conversation_id": conversation_id,
                    "created_at": created_at,
                    "item_data": item_dict,
                }

                item_records.append(item_record)

            await self.sql_store.insert(table="conversation_items", data=item_records)

        conversation = Conversation(
            id=conversation_id,
            created_at=created_at,
            metadata=request.metadata,
            object="conversation",
        )

        logger.debug("Created conversation", conversation_id=conversation_id)
        return conversation

    async def get_conversation(self, request: GetConversationRequest) -> Conversation:
        """Get a conversation with the given ID."""
        self._validate_conversation_id(request.conversation_id)
        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": request.conversation_id})

        if record is None:
            raise ConversationNotFoundError(request.conversation_id)

        return Conversation(
            id=record["id"], created_at=record["created_at"], metadata=record.get("metadata"), object="conversation"
        )

    async def update_conversation(self, conversation_id: str, request: UpdateConversationRequest) -> Conversation:
        """Update a conversation's metadata with the given ID"""
        self._validate_conversation_id(conversation_id)

        # verify conversation exists and trigger ABAC check before updating
        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": conversation_id})
        if record is None:
            raise ConversationNotFoundError(conversation_id)

        await self.sql_store.update(
            table="openai_conversations", data={"metadata": request.metadata}, where={"id": conversation_id}
        )

        return await self.get_conversation(GetConversationRequest(conversation_id=conversation_id))

    async def openai_delete_conversation(self, request: DeleteConversationRequest) -> ConversationDeletedResource:
        """Delete a conversation with the given ID."""
        self._validate_conversation_id(request.conversation_id)

        record = await self.sql_store.fetch_one(table="openai_conversations", where={"id": request.conversation_id})
        if record is None:
            raise ConversationNotFoundError(request.conversation_id)

        await self.sql_store.delete(table="openai_conversations", where={"id": request.conversation_id})

        logger.debug("Deleted conversation", conversation_id=request.conversation_id)
        return ConversationDeletedResource(id=request.conversation_id)

    def _validate_conversation_id(self, conversation_id: str) -> None:
        """Validate conversation ID format matches ``conv_`` + 48 hex chars."""
        if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
            raise InvalidParameterError(
                "conversation_id",
                conversation_id,
                "Conversation ID must match format 'conv_' followed by 48 lowercase hex characters.",
            )

    def _get_or_generate_item_id(self, item: ConversationItem, item_dict: dict[str, Any]) -> str:
        """Get existing item ID or generate one if missing."""
        if item.id is None:
            random_bytes = secrets.token_bytes(24)
            if item.type == "message":
                item_id = f"msg_{random_bytes.hex()}"
            else:
                item_id = f"item_{random_bytes.hex()}"
            item_dict["id"] = item_id
            return item_id
        return item.id

    async def _get_validated_conversation(self, conversation_id: str) -> Conversation:
        """Validate conversation ID format and return the conversation if it exists."""
        return await self.get_conversation(GetConversationRequest(conversation_id=conversation_id))

    async def add_items(self, conversation_id: str, request: AddItemsRequest) -> ConversationItemList:
        """Create (add) items to a conversation."""
        await self._get_validated_conversation(conversation_id)

        created_items = []
        base_time = int(time.time())

        for i, item in enumerate(request.items):
            item_dict = item.model_dump()
            item_id = self._get_or_generate_item_id(item, item_dict)

            # make each timestamp unique to maintain order
            created_at = base_time + i

            item_record = {
                "id": item_id,
                "conversation_id": conversation_id,
                "created_at": created_at,
                "item_data": item_dict,
            }

            await self.sql_store.upsert(
                table="conversation_items",
                data=item_record,
                conflict_columns=["id"],
            )

            created_items.append(item_dict)

        logger.debug(
            "Created items in conversation", created_items_count=len(created_items), conversation_id=conversation_id
        )

        # Convert created items (dicts) to proper ConversationItem types
        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [adapter.validate_python(item_dict) for item_dict in created_items]

        return ConversationItemList(
            data=response_items,
            first_id=created_items[0]["id"] if created_items else None,
            last_id=created_items[-1]["id"] if created_items else None,
            has_more=False,
        )

    async def retrieve(self, request: RetrieveItemRequest) -> ConversationItem:
        """Retrieve a conversation item."""
        self._validate_conversation_id(request.conversation_id)
        if not request.item_id:
            raise InvalidParameterError("item_id", request.item_id, "Must be a non-empty string.")

        # Get item from conversation_items table
        record = await self.sql_store.fetch_one(
            table="conversation_items", where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        if record is None:
            raise ConversationItemNotFoundError(request.item_id, request.conversation_id)

        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        return adapter.validate_python(record["item_data"])

    async def list_items(self, request: ListItemsRequest) -> ConversationItemList:
        """List items in the conversation."""
        # get_conversation validates the ID format and checks existence
        await self.get_conversation(GetConversationRequest(conversation_id=request.conversation_id))

        result = await self.sql_store.fetch_all(
            table="conversation_items", where={"conversation_id": request.conversation_id}
        )
        records = result.data

        if request.order is not None and request.order == "asc":
            records.sort(key=lambda x: x["created_at"])
        else:
            records.sort(key=lambda x: x["created_at"], reverse=True)

        actual_limit = request.limit or 20

        records = records[:actual_limit]
        items = [record["item_data"] for record in records]

        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [adapter.validate_python(item) for item in items]

        first_id = response_items[0].id if response_items else None
        last_id = response_items[-1].id if response_items else None

        return ConversationItemList(
            data=response_items,
            first_id=first_id,
            last_id=last_id,
            has_more=False,
        )

    async def openai_delete_conversation_item(self, request: DeleteItemRequest) -> ConversationItemDeletedResource:
        """Delete a conversation item."""
        if not request.item_id:
            raise InvalidParameterError("item_id", request.item_id, "Must be a non-empty string.")

        # _get_validated_conversation validates ID format and checks existence
        _ = await self._get_validated_conversation(request.conversation_id)

        record = await self.sql_store.fetch_one(
            table="conversation_items", where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        if record is None:
            raise ConversationItemNotFoundError(request.item_id, request.conversation_id)

        await self.sql_store.delete(
            table="conversation_items", where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        logger.debug("Deleted item from conversation", item_id=request.item_id, conversation_id=request.conversation_id)
        return ConversationItemDeletedResource(id=request.item_id)

    async def shutdown(self) -> None:
        pass
