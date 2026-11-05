# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import secrets
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, TypeAdapter
from sqlalchemy.exc import DBAPIError

from ogx.core.access_control.datatypes import AccessRule
from ogx.core.conversations.item_sync import ConversationItemSync
from ogx.core.conversations.validation import CONVERSATION_ID_PATTERN
from ogx.core.datatypes import StackConfig
from ogx.core.storage.sqlstore.authorized_sqlstore import authorized_sqlstore
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
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, DeleteOperation, SqlStore

logger = get_logger(name=__name__, category="openai_conversations")

# The name is fixed: access policies match sql_record resources on the physical table name.
ITEMS_TABLE = "conversation_items"
# Where the upgrade keeps the earlier table, which was keyed on id alone so an item id could
# belong to only one conversation. It is retained and not read; drop it manually once verified.
ITEMS_V1_TABLE = "conversation_items_v1"
ITEM_KEY_COLUMNS = ["conversation_id", "id"]
MIGRATIONS_TABLE = "conversation_migrations"
ITEMS_BACKFILL_KEY = "conversation_items_backfill:v2"
# Several workers can start migrating the same database at once on the first upgrade. The migration
# is idempotent, so a worker that loses a DDL race to a sibling retries and settles.
_MIGRATION_ATTEMPTS = 5
_MIGRATION_RETRY_DELAY_SECONDS = 0.1


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


class ConversationServiceImpl(Conversations, ConversationItemSync):
    """Built-in conversation service implementation using AuthorizedSqlStore."""

    def __init__(self, config: ConversationServiceConfig, deps: dict[Api, Any]):
        self.config = config
        self.deps = deps
        self.policy = config.policy

        # Use conversations store reference from run config
        conversations_ref = config.config.storage.stores.conversations
        if not conversations_ref:
            raise ServiceNotEnabledError("storage.stores.conversations")

        self._conversations_ref = conversations_ref

    async def initialize(self) -> None:
        """Initialize the store and create tables."""
        self.sql_store = await authorized_sqlstore(self._conversations_ref, self.policy)
        await self.sql_store.create_table(
            "openai_conversations",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "items": ColumnType.JSON,  # Deprecated: kept for backward compatibility, use ITEMS_TABLE instead
                "metadata": ColumnType.JSON,
            },
        )

        await self._create_items_table()

    async def _create_items_table(self) -> None:
        """Create ITEMS_TABLE keyed on (conversation_id, id), migrating an id-keyed one in place.

        A table left by an earlier server is recognised by its primary key, renamed to
        ITEMS_V1_TABLE and recreated under the same name, so access policies that name
        the table keep matching. Its rows are then copied across once.

        Several workers can start migrating the same database at once on the first upgrade. Every
        step is idempotent, so a worker that loses a DDL race to a sibling (the rename, or the
        recreate of the live table) simply retries and re-reads the now-migrated state instead of
        failing its boot.
        """
        store = self.sql_store.sql_store
        for attempt in range(_MIGRATION_ATTEMPTS):
            try:
                await self._rename_id_keyed_table_if_present(store)
                await self.sql_store.create_table(
                    ITEMS_TABLE,
                    {
                        "conversation_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                        "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                        "created_at": ColumnType.INTEGER,
                        "sort_order": ColumnType.INTEGER,
                        "item_data": ColumnType.JSON,
                        # Declared explicitly so the backfill below carries it even when tenancy is disabled.
                        "tenant_id": ColumnType.STRING,
                    },
                )
                await self._backfill_v1_items()
                return
            except DBAPIError:
                if attempt == _MIGRATION_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_MIGRATION_RETRY_DELAY_SECONDS)

    async def _rename_id_keyed_table_if_present(self, store: SqlStore) -> None:
        """Rename an id-keyed ITEMS_TABLE to ITEMS_V1_TABLE if (and only if) it still is one.

        A fresh or already-composite table needs no rename. A rename that loses the race to a
        sibling raises a database error; the caller retries the idempotent migration and settles.
        An ITEMS_V1_TABLE that is not a sibling's work, while the live table is still id-keyed,
        makes every rename fail, so the boot fails rather than silently skipping the migration.
        """
        primary_key = await store.primary_key_columns(ITEMS_TABLE)
        # Compare as a set: the database may report composite primary key columns in any order.
        if primary_key is None or set(primary_key) == set(ITEM_KEY_COLUMNS):
            return
        await store.rename_table(ITEMS_TABLE, ITEMS_V1_TABLE)
        logger.info("Renamed the id-keyed conversation items table", table=ITEMS_TABLE, renamed_to=ITEMS_V1_TABLE)

    async def _backfill_v1_items(self) -> None:
        """Copy the rows of ITEMS_V1_TABLE into ITEMS_TABLE, once per database.

        Nothing runs on a database that never had the v1 table. While it exists, a
        completion flag in MIGRATIONS_TABLE keeps later boots from scanning it again,
        and a warning reminds the operator that it is still there.
        """
        store = self.sql_store.sql_store
        if not await store.table_exists(ITEMS_V1_TABLE):
            return
        await store.create_table(
            MIGRATIONS_TABLE,
            {
                "name": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "completed_at": ColumnType.INTEGER,
            },
        )
        if await store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY}) is None:
            copied = await store.copy_missing_rows(ITEMS_V1_TABLE, ITEMS_TABLE, ITEM_KEY_COLUMNS)
            # do-nothing so a sibling that sets the flag first does not fail this worker.
            await store.insert_do_nothing(
                MIGRATIONS_TABLE, {"name": ITEMS_BACKFILL_KEY, "completed_at": int(time.time())}
            )
            logger.info(
                "Backfilled conversation items from the id-keyed table",
                copied_rows=copied,
                source_table=ITEMS_V1_TABLE,
                table=ITEMS_TABLE,
            )
        logger.warning(
            "Pre-migration conversation items table retained; verify the backfill and drop it manually once confirmed",
            retained_table=ITEMS_V1_TABLE,
            table=ITEMS_TABLE,
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

        item_records = await self._resolve_item_records(conversation_id, request.items or [], created_at, 0)

        await self.sql_store.insert(
            table="openai_conversations",
            data=record_data,
        )

        if item_records:
            await self.sql_store.insert(table=ITEMS_TABLE, data=item_records)

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

        await self.sql_store.delete_many(
            [
                DeleteOperation(
                    table=ITEMS_TABLE,
                    where={"conversation_id": request.conversation_id},
                ),
                DeleteOperation(
                    table="openai_conversations",
                    where={"id": request.conversation_id},
                ),
            ]
        )

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

    def _generate_item_id(self, item: ConversationItem) -> str:
        random_bytes = secrets.token_bytes(24)
        if item.type == "message":
            return f"msg_{random_bytes.hex()}"
        return f"item_{random_bytes.hex()}"

    async def _get_validated_conversation(self, conversation_id: str) -> Conversation:
        """Validate conversation ID format and return the conversation if it exists."""
        return await self.get_conversation(GetConversationRequest(conversation_id=conversation_id))

    async def _item_in_conversation(self, item_id: str, conversation_id: str) -> bool:
        record = await self.sql_store.fetch_one(
            table=ITEMS_TABLE, where={"id": item_id, "conversation_id": conversation_id}
        )
        return record is not None

    async def _resolve_item_records(
        self,
        conversation_id: str,
        items: Sequence[ConversationItem],
        base_time: int,
        base_sort_order: int,
    ) -> list[dict[str, Any]]:
        """Build the rows to insert, resolving client-supplied ids the way the OpenAI API does.

        An id already in the target conversation is rejected. An id of an item the caller
        can read in another conversation copies that item with its original content, and
        the content sent for it is ignored. Any other id is replaced by a server-minted
        one. Nothing is written here, so a rejected batch leaves no partial rows.
        """
        item_records: list[dict[str, Any]] = []
        copied_ids: set[str] = set()
        for i, item in enumerate(items):
            item_dict = item.model_dump()
            if item.id is None:
                item_id = self._generate_item_id(item)
                item_dict["id"] = item_id
            else:
                if item.id in copied_ids or await self._item_in_conversation(item.id, conversation_id):
                    raise InvalidParameterError("items", item.id, "Item already in conversation.")
                source = await self.sql_store.fetch_one(table=ITEMS_TABLE, where={"id": item.id})
                if source is None:
                    item_id = self._generate_item_id(item)
                    item_dict["id"] = item_id
                else:
                    item_id = item.id
                    item_dict = source["item_data"]
                    copied_ids.add(item_id)

            item_records.append(
                {
                    "id": item_id,
                    "conversation_id": conversation_id,
                    "created_at": base_time,
                    "sort_order": base_sort_order + i,
                    "item_data": item_dict,
                }
            )
        return item_records

    def _item_list(self, item_records: Sequence[dict[str, Any]]) -> ConversationItemList:
        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [
            adapter.validate_python(item_record["item_data"]) for item_record in item_records
        ]
        return ConversationItemList(
            data=response_items,
            first_id=response_items[0].id if response_items else "",
            last_id=response_items[-1].id if response_items else "",
            has_more=False,
        )

    async def _next_sort_order(self, conversation_id: str) -> int:
        result = await self.sql_store.fetch_all(
            table=ITEMS_TABLE,
            where={"conversation_id": conversation_id},
            order_by=[("sort_order", "desc")],
            limit=1,
        )
        if result.data:
            current_max = result.data[0].get("sort_order")
            return (current_max + 1) if current_max is not None else 0
        return 0

    async def add_items(self, conversation_id: str, request: AddItemsRequest) -> ConversationItemList:
        """Create (add) items to a conversation."""
        await self._get_validated_conversation(conversation_id)

        base_time = int(time.time())
        base_sort_order = await self._next_sort_order(conversation_id)
        item_records = await self._resolve_item_records(conversation_id, request.items, base_time, base_sort_order)

        if item_records:
            await self.sql_store.insert(table=ITEMS_TABLE, data=item_records)

        logger.debug(
            "Created items in conversation", created_items_count=len(item_records), conversation_id=conversation_id
        )
        return self._item_list(item_records)

    async def sync_items(self, conversation_id: str, request: AddItemsRequest) -> None:
        """Append items to a conversation keeping their ids, skipping ids already present.

        Implements the internal ``ConversationItemSync`` protocol used by the Responses
        API to mirror a response into its conversation, where the output items must keep
        the ids the response reported and re-sent input items must not be duplicated or
        rejected. It is not an HTTP route.
        """
        await self._get_validated_conversation(conversation_id)

        base_time = int(time.time())
        base_sort_order = await self._next_sort_order(conversation_id)

        item_records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in request.items:
            item_dict = item.model_dump()
            item_id = item.id
            if item_id is None:
                item_id = self._generate_item_id(item)
                item_dict["id"] = item_id
            elif item_id in seen_ids or await self._item_in_conversation(item_id, conversation_id):
                continue
            seen_ids.add(item_id)
            item_records.append(
                {
                    "id": item_id,
                    "conversation_id": conversation_id,
                    "created_at": base_time,
                    "sort_order": base_sort_order + len(item_records),
                    "item_data": item_dict,
                }
            )

        if item_records:
            await self.sql_store.insert(table=ITEMS_TABLE, data=item_records)

        logger.debug(
            "Synced items to conversation", synced_items_count=len(item_records), conversation_id=conversation_id
        )

    async def retrieve(self, request: RetrieveItemRequest) -> ConversationItem:
        """Retrieve a conversation item."""
        self._validate_conversation_id(request.conversation_id)
        if not request.item_id:
            raise InvalidParameterError("item_id", request.item_id, "Must be a non-empty string.")

        await self._get_validated_conversation(request.conversation_id)

        # Get item from the items table
        record = await self.sql_store.fetch_one(
            table=ITEMS_TABLE, where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        if record is None:
            raise ConversationItemNotFoundError(request.item_id, request.conversation_id)

        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        return adapter.validate_python(record["item_data"])

    async def list_items(self, request: ListItemsRequest) -> ConversationItemList:
        """List items in the conversation with cursor pagination."""
        await self._get_validated_conversation(request.conversation_id)

        order = request.order if request.order is not None else "desc"
        limit = request.limit or 20

        if request.after:
            cursor_record = await self.sql_store.fetch_one(
                table=ITEMS_TABLE,
                where={"id": request.after, "conversation_id": request.conversation_id},
            )
            if cursor_record is None:
                raise ConversationItemNotFoundError(request.after, request.conversation_id)

        result = await self.sql_store.fetch_all(
            table=ITEMS_TABLE,
            where={"conversation_id": request.conversation_id},
            order_by=[("sort_order", order)],
            cursor=("id", request.after) if request.after else None,
            limit=limit,
        )

        adapter: TypeAdapter[ConversationItem] = TypeAdapter(ConversationItem)
        response_items: list[ConversationItem] = [
            adapter.validate_python(record["item_data"]) for record in result.data
        ]

        first_id = response_items[0].id if response_items else ""
        last_id = response_items[-1].id if response_items else ""

        return ConversationItemList(
            data=response_items,
            first_id=first_id,
            last_id=last_id,
            has_more=result.has_more,
        )

    async def openai_delete_conversation_item(self, request: DeleteItemRequest) -> Conversation:
        """Delete a conversation item and return the parent conversation."""
        if not request.item_id:
            raise InvalidParameterError("item_id", request.item_id, "Must be a non-empty string.")

        conversation = await self._get_validated_conversation(request.conversation_id)

        record = await self.sql_store.fetch_one(
            table=ITEMS_TABLE, where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        if record is None:
            raise ConversationItemNotFoundError(request.item_id, request.conversation_id)

        await self.sql_store.delete(
            table=ITEMS_TABLE, where={"id": request.item_id, "conversation_id": request.conversation_id}
        )

        logger.debug("Deleted item from conversation", item_id=request.item_id, conversation_id=request.conversation_id)
        return conversation

    async def shutdown(self) -> None:
        pass
