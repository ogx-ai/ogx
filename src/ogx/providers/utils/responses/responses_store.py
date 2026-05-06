# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.datatypes import AccessRule
from ogx.core.storage.datatypes import ResponsesStoreReference, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx.log import get_logger
from ogx_api import (
    InvalidParameterError,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseObject,
    OpenAIResponseObjectWithInput,
    Order,
    ResponseInputItemNotFoundError,
    ResponseItemInclude,
    ResponseNotFoundError,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="openai_responses")


class _OpenAIResponseObjectWithInputAndMessages(OpenAIResponseObjectWithInput):
    """Internal class for storing responses with chat completion messages.

    This extends the public OpenAIResponseObjectWithInput with messages field
    for internal storage. The messages field is not exposed in the public API.

    The messages field is optional for backward compatibility with responses
    stored before this feature was added.
    """

    messages: list[OpenAIMessageParam] | None = None


class ResponsesStore:
    """Persistent store for OpenAI Responses API objects with SQL-backed storage."""

    def __init__(
        self,
        reference: ResponsesStoreReference | SqlStoreReference,
        policy: list[AccessRule],
    ):
        if isinstance(reference, ResponsesStoreReference):
            self.reference = reference
        else:
            self.reference = ResponsesStoreReference(**reference.model_dump())

        self.policy = policy

    async def initialize(self):
        """Create the necessary tables if they don't exist."""
        base_store = sqlstore_impl(self.reference)
        self.sql_store = AuthorizedSqlStore(base_store, self.policy)

        await self.sql_store.create_table(
            self.reference.table_name,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "response_object": ColumnType.JSON,
                "model": ColumnType.STRING,
            },
        )

        await self.sql_store.create_table(
            "conversation_messages",
            {
                "conversation_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "messages": ColumnType.JSON,
            },
        )

    async def shutdown(self) -> None:
        return

    async def flush(self) -> None:
        """Maintained for compatibility; no-op now that writes are synchronous."""
        return

    async def store_response_object(
        self,
        response_object: OpenAIResponseObject,
        input: list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
    ) -> None:
        await self._write_response_object(response_object, input, messages)

    async def upsert_response_object(
        self,
        response_object: OpenAIResponseObject,
        input: list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
    ) -> None:
        """Upsert response object using INSERT on first call, UPDATE on subsequent calls.

        This method enables incremental persistence during streaming, allowing clients
        to poll GET /v1/responses/{response_id} and see in-progress turn state.

        :param response_object: The response object to store/update.
        :param input: The input items for the response.
        :param messages: The chat completion messages (for conversation continuity).
        """

        data = response_object.model_dump()
        data["input"] = [input_item.model_dump() for input_item in input]
        data["messages"] = [msg.model_dump() for msg in messages]

        await self.sql_store.upsert(
            table=self.reference.table_name,
            data={
                "id": data["id"],
                "created_at": data["created_at"],
                "model": data["model"],
                "response_object": data,
            },
            conflict_columns=["id"],
            update_columns=["response_object"],
        )

    async def _write_response_object(
        self,
        response_object: OpenAIResponseObject,
        input: list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
    ) -> None:
        data = response_object.model_dump()
        data["input"] = [input_item.model_dump() for input_item in input]
        data["messages"] = [msg.model_dump() for msg in messages]

        await self.sql_store.insert(
            self.reference.table_name,
            {
                "id": data["id"],
                "created_at": data["created_at"],
                "model": data["model"],
                "response_object": data,
            },
        )

    async def list_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        """
        List responses from the database.

        :param after: The ID of the last response to return.
        :param limit: The maximum number of responses to return.
        :param model: The model to filter by.
        :param order: The order to sort the responses by.
        """

        if not order:
            order = Order.desc

        where_conditions = {}
        if model:
            where_conditions["model"] = model

        paginated_result = await self.sql_store.fetch_all(
            table=self.reference.table_name,
            where=where_conditions if where_conditions else None,
            order_by=[("created_at", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        data = [OpenAIResponseObjectWithInput(**row["response_object"]) for row in paginated_result.data]
        return ListOpenAIResponseObject(
            data=data,
            has_more=paginated_result.has_more,
            first_id=data[0].id if data else "",
            last_id=data[-1].id if data else "",
        )

    async def get_response_object(self, response_id: str) -> _OpenAIResponseObjectWithInputAndMessages:
        """
        Get a response object with automatic access control checking.
        """

        row = await self.sql_store.fetch_one(
            self.reference.table_name,
            where={"id": response_id},
        )

        if not row:
            # SecureSqlStore will return None if record doesn't exist OR access is denied
            # This provides security by not revealing whether the record exists
            raise ResponseNotFoundError(response_id) from None

        return _OpenAIResponseObjectWithInputAndMessages(**row["response_object"])

    async def delete_response_object(self, response_id: str) -> OpenAIDeleteResponseObject:
        row = await self.sql_store.fetch_one(self.reference.table_name, where={"id": response_id})
        if not row:
            raise ResponseNotFoundError(response_id)
        await self.sql_store.delete(self.reference.table_name, where={"id": response_id})
        return OpenAIDeleteResponseObject(id=response_id)

    async def update_response_object(
        self,
        response_object: OpenAIResponseObject,
        input: list[OpenAIResponseInput] | None = None,
    ) -> None:
        """Update an existing response object in storage.

        :param response_object: The updated response object.
        :param input: Optional input items (if None, existing input is preserved).
        """
        # Fetch existing data to preserve input/messages if not provided
        existing_row = await self.sql_store.fetch_one(
            self.reference.table_name,
            where={"id": response_object.id},
        )

        if not existing_row:
            logger.critical(
                "Response not found during update - this should never happen", response_id=response_object.id
            )
            raise RuntimeError(f"Response with id {response_object.id} not found during update")

        existing_data = existing_row["response_object"]

        data = response_object.model_dump()
        # Preserve existing input if not provided
        if input is not None:
            data["input"] = [input_item.model_dump() for input_item in input]
        else:
            data["input"] = existing_data.get("input", [])
        # Messages are stored in the blob by store/upsert_response_object.
        # Preserve them here so updating status doesn't clobber them.
        data["messages"] = existing_data.get("messages", [])

        await self.sql_store.update(
            self.reference.table_name,
            data={
                "created_at": data["created_at"],
                "model": data["model"],
                "response_object": data,
            },
            where={"id": response_object.id},
        )

    async def list_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[ResponseItemInclude] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """
        List input items for a given response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        """
        if include:
            raise NotImplementedError("Include is not supported yet")
        if before and after:
            raise InvalidParameterError(
                "before/after",
                f"before={before!r}, after={after!r}",
                "Cannot specify both 'before' and 'after' parameters",
            )

        response_with_input_and_messages = await self.get_response_object(response_id)
        # Filter out compaction items (matching OpenAI behavior: input_items hides compaction)
        items = [
            item
            for item in response_with_input_and_messages.input
            if not (hasattr(item, "type") and getattr(item, "type", None) == "compaction")
        ]

        if order == Order.desc:
            items = list(reversed(items))

        start_index = 0
        end_index = len(items)

        if after or before:
            for i, item in enumerate(items):
                item_id = getattr(item, "id", None)
                if after and item_id == after:
                    start_index = i + 1
                if before and item_id == before:
                    end_index = i
                    break

            if after and start_index == 0:
                raise ResponseInputItemNotFoundError(after, response_id)
            if before and end_index == len(items):
                raise ResponseInputItemNotFoundError(before, response_id)

        items = items[start_index:end_index]

        # Apply limit
        if limit is not None:
            items = items[:limit]

        return ListOpenAIResponseInputItem(data=items)

    async def store_conversation_messages(self, conversation_id: str, messages: list[OpenAIMessageParam]) -> None:
        """Store messages for a conversation.

        :param conversation_id: The conversation identifier.
        :param messages: List of OpenAI message parameters to store.
        """

        # Serialize messages to dict format for JSON storage
        messages_data = [msg.model_dump() for msg in messages]

        await self.sql_store.upsert(
            table="conversation_messages",
            data={"conversation_id": conversation_id, "messages": messages_data},
            conflict_columns=["conversation_id"],
            update_columns=["messages"],
        )

        logger.debug("Stored messages for conversation", messages_count=len(messages), conversation_id=conversation_id)

    async def get_conversation_messages(self, conversation_id: str) -> list[OpenAIMessageParam] | None:
        """Get stored messages for a conversation.

        :param conversation_id: The conversation identifier.
        :returns: List of OpenAI message parameters, or None if no messages stored.
        """

        record = await self.sql_store.fetch_one(
            table="conversation_messages",
            where={"conversation_id": conversation_id},
        )

        if record is None:
            return None

        # Deserialize messages from JSON storage
        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[OpenAIMessageParam])
        return adapter.validate_python(record["messages"])
