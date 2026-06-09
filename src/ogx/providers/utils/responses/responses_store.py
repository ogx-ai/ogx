# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Sequence

from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import AccessRule
from ogx.core.storage.datatypes import ResponsesStoreReference, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import authorized_sqlstore
from ogx.log import get_logger
from ogx_api import (
    InvalidParameterError,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectWithInput,
    OpenAIResponseOutputMessageContent,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageReasoningItem,
    Order,
    ResponseInputItemNotFoundError,
    ResponseItemInclude,
    ResponseNotFoundError,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="openai_responses")


def _filter_message_include_fields(
    item: OpenAIResponseMessage,
    include_values: set[str],
) -> OpenAIResponseMessage:
    if isinstance(item.content, str):
        return item

    filtered_content: list[OpenAIResponseInputMessageContent | OpenAIResponseOutputMessageContent] = []
    include_input_image_url = ResponseItemInclude.message_input_image_image_url.value in include_values
    include_output_text_logprobs = ResponseItemInclude.message_output_text_logprobs.value in include_values
    item_changed = False

    for content_item in item.content:
        filtered_content_item = content_item

        if isinstance(content_item, OpenAIResponseInputMessageContentImage) and not include_input_image_url:
            if content_item.image_url is not None:
                filtered_content_item = content_item.model_copy(update={"image_url": None})
                item_changed = True
        elif (
            isinstance(content_item, OpenAIResponseOutputMessageContentOutputText) and not include_output_text_logprobs
        ):
            if content_item.logprobs is not None:
                filtered_content_item = content_item.model_copy(update={"logprobs": None})
                item_changed = True

        filtered_content.append(filtered_content_item)

    if not item_changed:
        return item

    return item.model_copy(update={"content": filtered_content})


def _apply_include_filter(
    items: list[OpenAIResponseInput],
    include: list[ResponseItemInclude] | None,
) -> list[OpenAIResponseInput]:
    include_values = {str(value) for value in include or []}
    include_file_search_results = ResponseItemInclude.file_search_call_results.value in include_values
    include_reasoning_content = ResponseItemInclude.reasoning_encrypted_content.value in include_values
    filtered_items: list[OpenAIResponseInput] = []

    for item in items:
        if isinstance(item, OpenAIResponseOutputMessageFileSearchToolCall):
            if item.results is not None and not include_file_search_results:
                filtered_items.append(item.model_copy(update={"results": None}))
            else:
                filtered_items.append(item)
        elif isinstance(item, OpenAIResponseOutputMessageReasoningItem):
            if item.content is not None and not include_reasoning_content:
                filtered_items.append(item.model_copy(update={"content": None}))
            else:
                filtered_items.append(item)
        elif isinstance(item, OpenAIResponseMessage):
            filtered_items.append(_filter_message_include_fields(item, include_values))
        else:
            filtered_items.append(item)

    return filtered_items


class _OpenAIResponseObjectWithInputAndMessages(OpenAIResponseObjectWithInput):
    """Internal class for storing responses with chat completion messages.

    This extends the public OpenAIResponseObjectWithInput with messages field
    for internal storage. The messages field is not exposed in the public API.

    The messages field is optional for backward compatibility with responses
    stored before this feature was added.
    """

    messages: list[OpenAIMessageParam] | None = None
    input_storage_mode: str | None = None


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
        self.sql_store = await authorized_sqlstore(self.reference, self.policy)

        await self.sql_store.create_table(
            self.reference.table_name,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "response_object": ColumnType.JSON,
                "model": ColumnType.STRING,
                "previous_response_id": ColumnType.STRING,
                "input_storage_mode": ColumnType.STRING,
            },
        )
        # Backward-compatible schema migration for existing stores.
        await self.sql_store.add_column_if_not_exists(
            self.reference.table_name,
            "previous_response_id",
            ColumnType.STRING,
        )
        await self.sql_store.add_column_if_not_exists(
            self.reference.table_name,
            "input_storage_mode",
            ColumnType.STRING,
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
        incremental_input: bool = False,
    ) -> None:
        await self._write_response_object(response_object, input, messages, incremental_input)

    async def upsert_response_object(
        self,
        response_object: OpenAIResponseObject,
        input: list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
        incremental_input: bool = False,
    ) -> None:
        """Upsert response object using INSERT on first call, UPDATE on subsequent calls.

        This method enables incremental persistence during streaming, allowing clients
        to poll GET /v1/responses/{response_id} and see in-progress turn state.

        :param response_object: The response object to store/update.
        :param input: The input items for the response.
        :param messages: The chat completion messages (for conversation continuity).
        :param incremental_input: If True, input contains only new items for this turn.
        """

        data = response_object.model_dump()
        data["input"] = [input_item.model_dump() for input_item in input]
        data["messages"] = [msg.model_dump() for msg in messages]

        previous_response_id = data.get("previous_response_id")
        storage_mode = "incremental" if incremental_input else None
        preserve_materialized_snapshot = False

        if storage_mode:
            data["input_storage_mode"] = storage_mode
            # If the row was previously materialized due to parent deletion, keep that
            # full snapshot across subsequent streaming upserts.
            existing_row = await self.sql_store.fetch_one(
                self.reference.table_name,
                where={"id": data["id"]},
            )
            if existing_row:
                existing_data = existing_row["response_object"]
                existing_previous_response_id = existing_row.get("previous_response_id")
                existing_storage_mode = existing_row.get("input_storage_mode")
                if (
                    existing_previous_response_id is None
                    and existing_storage_mode is None
                    and previous_response_id is not None
                ):
                    preserve_materialized_snapshot = True
                    data["input"] = existing_data.get("input", [])
                    data["previous_response_id"] = None
                    data.pop("input_storage_mode", None)
                    previous_response_id = None
                    storage_mode = None
                elif previous_response_id is not None and existing_previous_response_id != previous_response_id:
                    # Parent deletion can rewrite ancestry for in-progress children.
                    # If a stale writer still sends the old previous_response_id, keep
                    # the rewritten chain to avoid re-introducing dangling ancestry.
                    preserve_materialized_snapshot = True
                    data["input"] = existing_data.get("input", [])
                    data["previous_response_id"] = existing_previous_response_id
                    previous_response_id = existing_previous_response_id
                    storage_mode = existing_storage_mode
                    if storage_mode is not None:
                        data["input_storage_mode"] = storage_mode
                    else:
                        data.pop("input_storage_mode", None)
        else:
            data.pop("input_storage_mode", None)

        await self.sql_store.upsert(
            table=self.reference.table_name,
            data={
                "id": data["id"],
                "created_at": data["created_at"],
                "model": data["model"],
                "previous_response_id": previous_response_id,
                "input_storage_mode": storage_mode,
                "response_object": data,
            },
            conflict_columns=["id"],
            update_columns=[
                "created_at",
                "model",
                "previous_response_id",
                "input_storage_mode",
                "response_object",
            ],
        )
        if preserve_materialized_snapshot:
            logger.debug("Preserved materialized snapshot during incremental upsert", response_id=data["id"])

    async def _write_response_object(
        self,
        response_object: OpenAIResponseObject,
        input: list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
        incremental_input: bool = False,
    ) -> None:
        data = response_object.model_dump()
        data["input"] = [input_item.model_dump() for input_item in input]
        data["messages"] = [msg.model_dump() for msg in messages]
        storage_mode = "incremental" if incremental_input else None
        if storage_mode:
            data["input_storage_mode"] = storage_mode
        else:
            data.pop("input_storage_mode", None)
        previous_response_id = data.get("previous_response_id")

        await self.sql_store.insert(
            self.reference.table_name,
            {
                "id": data["id"],
                "created_at": data["created_at"],
                "model": data["model"],
                "previous_response_id": previous_response_id,
                "input_storage_mode": storage_mode,
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

        response_cache: dict[str, _OpenAIResponseObjectWithInputAndMessages | None] = {}
        ordered_responses: list[_OpenAIResponseObjectWithInputAndMessages] = []
        for row in paginated_result.data:
            response = _OpenAIResponseObjectWithInputAndMessages(**row["response_object"])
            response_cache[response.id] = response
            ordered_responses.append(response)

        full_input_cache: dict[str, Sequence[OpenAIResponseInput]] = {}
        data: list[OpenAIResponseObjectWithInput] = []
        active_response_ids: set[str] = set()
        for response in ordered_responses:
            if self._is_incremental_response(response) and response.previous_response_id:
                response.input = await self._reconstruct_full_input_with_cache(
                    current_response=response,
                    response_cache=response_cache,
                    full_input_cache=full_input_cache,
                    active_response_ids=active_response_ids,
                )

            data.append(OpenAIResponseObjectWithInput(**response.model_dump()))

        return ListOpenAIResponseObject(
            data=data,
            has_more=paginated_result.has_more,
            first_id=data[0].id if data else "",
            last_id=data[-1].id if data else "",
        )

    async def get_response_object(
        self, response_id: str, reconstruct_input: bool = True
    ) -> _OpenAIResponseObjectWithInputAndMessages:
        """Get a response object with automatic access control checking.

        :param response_id: The ID of the response to retrieve.
        :param reconstruct_input: If True (default), reconstruct full input chain for
            responses stored in incremental mode. Set to False when only checking
            response status to avoid unnecessary DB queries.
        """

        row = await self.sql_store.fetch_one(
            self.reference.table_name,
            where={"id": response_id},
        )

        if not row:
            # SecureSqlStore will return None if record doesn't exist OR access is denied
            # This provides security by not revealing whether the record exists
            raise ResponseNotFoundError(response_id) from None

        response_data = row["response_object"]
        response = _OpenAIResponseObjectWithInputAndMessages(**response_data)

        if reconstruct_input and self._is_incremental_response(response) and response.previous_response_id:
            response.input = await self._reconstruct_full_input(response)

        return response

    @staticmethod
    def _is_incremental_response(response: _OpenAIResponseObjectWithInputAndMessages) -> bool:
        return response.input_storage_mode == "incremental"

    async def _fetch_response_for_reconstruction(
        self,
        response_id: str,
        response_cache: dict[str, _OpenAIResponseObjectWithInputAndMessages | None],
    ) -> _OpenAIResponseObjectWithInputAndMessages | None:
        if response_id in response_cache:
            return response_cache[response_id]

        row = await self.sql_store.fetch_one(
            self.reference.table_name,
            where={"id": response_id},
        )
        if not row:
            response_cache[response_id] = None
            return None

        response = _OpenAIResponseObjectWithInputAndMessages(**row["response_object"])
        response_cache[response_id] = response
        return response

    async def _reconstruct_full_input_with_cache(
        self,
        current_response: _OpenAIResponseObjectWithInputAndMessages,
        response_cache: dict[str, _OpenAIResponseObjectWithInputAndMessages | None],
        full_input_cache: dict[str, Sequence[OpenAIResponseInput]],
        active_response_ids: set[str],
    ) -> Sequence[OpenAIResponseInput]:
        cached_input = full_input_cache.get(current_response.id)
        if cached_input is not None:
            return cached_input

        chain: list[_OpenAIResponseObjectWithInputAndMessages] = []
        response = current_response
        added_active_ids: list[str] = []
        try:
            while True:
                cached_chain_input = full_input_cache.get(response.id)
                if cached_chain_input is not None:
                    break

                if response.id in active_response_ids:
                    logger.warning(
                        "Detected cycle in response ancestry chain; using stored input",
                        response_id=response.id,
                    )
                    return list(current_response.input)

                active_response_ids.add(response.id)
                added_active_ids.append(response.id)
                chain.append(response)

                if not (self._is_incremental_response(response) and response.previous_response_id):
                    break

                previous_response = await self._fetch_response_for_reconstruction(
                    response.previous_response_id,
                    response_cache,
                )
                if previous_response is None:
                    break

                response = previous_response

            anchor_response_id = response.id
            anchor_input = full_input_cache.get(anchor_response_id)
            if anchor_input is None:
                anchor_input = list(response.input)
                full_input_cache[anchor_response_id] = anchor_input

            for chain_response in reversed(chain):
                if chain_response.id == anchor_response_id:
                    full_input_cache[chain_response.id] = list(anchor_input)
                    continue

                if self._is_incremental_response(chain_response) and chain_response.previous_response_id:
                    previous_response = await self._fetch_response_for_reconstruction(
                        chain_response.previous_response_id,
                        response_cache,
                    )
                    previous_input = full_input_cache.get(chain_response.previous_response_id)
                    if previous_response and previous_input is not None:
                        full_input = list(previous_input)
                        full_input.extend(previous_response.output)
                        full_input.extend(chain_response.input)
                        full_input_cache[chain_response.id] = full_input
                        continue

                full_input_cache[chain_response.id] = list(chain_response.input)

            return full_input_cache.get(current_response.id, list(current_response.input))
        finally:
            for response_id in added_active_ids:
                active_response_ids.discard(response_id)

    async def _reconstruct_full_input(
        self,
        current_response: _OpenAIResponseObjectWithInputAndMessages,
    ) -> Sequence[OpenAIResponseInput]:
        """Reconstruct full accumulated input for incremental responses."""
        response_cache: dict[str, _OpenAIResponseObjectWithInputAndMessages | None] = {
            current_response.id: current_response
        }
        return await self._reconstruct_full_input_with_cache(
            current_response=current_response,
            response_cache=response_cache,
            full_input_cache={},
            active_response_ids=set(),
        )

    async def _materialize_incremental_children(
        self,
        parent_response: _OpenAIResponseObjectWithInputAndMessages,
    ) -> None:
        """Rewrite incremental direct children before parent deletion.

        Each child currently references `parent_response.id`. Before deleting that row,
        rewrite the child to bypass the soon-to-be-missing parent while preserving chain
        semantics:
        - child.input = parent.input + parent.output + child.input
        - child.previous_response_id = parent.previous_response_id (if parent is incremental)
        - otherwise child becomes a materialized snapshot (previous_response_id=None)
        """
        parent_response_id = parent_response.id

        # Use the underlying SQL store so children hidden by READ policy are still
        # materialized before parent deletion.
        rows = await self.sql_store.sql_store.fetch_all(
            table=self.reference.table_name,
            where={"previous_response_id": parent_response_id},
        )

        for row in rows.data:
            response_data = row["response_object"]
            if response_data.get("input_storage_mode") != "incremental":
                continue

            child_response = _OpenAIResponseObjectWithInputAndMessages(**response_data)
            rewritten_input: list[OpenAIResponseInput] = list(parent_response.input)
            rewritten_input.extend(parent_response.output)
            rewritten_input.extend(child_response.input)

            child_data = child_response.model_dump()
            child_data["input"] = [input_item.model_dump() for input_item in rewritten_input]
            child_data["messages"] = response_data.get("messages", [])

            if self._is_incremental_response(parent_response) and parent_response.previous_response_id:
                child_previous_response_id = parent_response.previous_response_id
                child_input_storage_mode = "incremental"
                child_data["previous_response_id"] = child_previous_response_id
                child_data["input_storage_mode"] = child_input_storage_mode
            else:
                child_previous_response_id = None
                child_input_storage_mode = None
                child_data["previous_response_id"] = None
                child_data.pop("input_storage_mode", None)

            # This write is an internal side effect of deleting the parent response.
            # It must not require UPDATE permission on child rows.
            await self.sql_store.sql_store.update(
                self.reference.table_name,
                data={
                    "created_at": child_data["created_at"],
                    "model": child_data["model"],
                    "previous_response_id": child_previous_response_id,
                    "input_storage_mode": child_input_storage_mode,
                    "response_object": child_data,
                },
                where={"id": child_response.id},
            )

    async def delete_response_object(self, response_id: str) -> OpenAIDeleteResponseObject:
        row = await self.sql_store.fetch_one(self.reference.table_name, where={"id": response_id})
        if not row:
            raise ResponseNotFoundError(response_id)

        parent_response = _OpenAIResponseObjectWithInputAndMessages(**row["response_object"])

        # Ensure delete access before running any internal materialization writes.
        await self.sql_store.check_access_for_rows(
            table=self.reference.table_name,
            where={"id": response_id},
            action=Action.DELETE,
        )

        # Prevent descendant input truncation after ancestor deletion by rewriting
        # incremental direct children to bypass this parent record.
        await self._materialize_incremental_children(parent_response)

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
        # Preserve incremental input metadata so chained reconstruction remains available.
        if "input_storage_mode" in existing_data:
            data["input_storage_mode"] = existing_data["input_storage_mode"]
        # Status updates should not mutate ancestry linkage.
        if "previous_response_id" in existing_data:
            data["previous_response_id"] = existing_data["previous_response_id"]

        await self.sql_store.update(
            self.reference.table_name,
            data={
                "created_at": data["created_at"],
                "model": data["model"],
                "previous_response_id": data.get("previous_response_id"),
                "input_storage_mode": data.get("input_storage_mode"),
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

        items = _apply_include_filter(items, include)

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
