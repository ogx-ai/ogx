# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
from typing import Any, NamedTuple

from sqlalchemy.exc import IntegrityError

from ogx.core.datatypes import AccessRule
from ogx.core.storage.datatypes import InferenceStoreReference, StorageBackendType
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import _SQLSTORE_BACKENDS, sqlstore_impl
from ogx.core.task import (
    RequestContext,
    activate_request_context,
    capture_request_context,
    create_detached_background_task,
)
from ogx.log import get_logger
from ogx_api import (
    ListOpenAIChatCompletionResponse,
    OpenAIChatCompletion,
    OpenAICompletionWithInputMessages,
    OpenAIMessageParam,
    Order,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="inference")


class _WriteItem(NamedTuple):
    completion: OpenAIChatCompletion
    messages: list[OpenAIMessageParam]
    request_context: RequestContext


class InferenceStore:
    """Persistent store for chat completion records with async write queue support."""

    def __init__(
        self,
        reference: InferenceStoreReference,
        policy: list[AccessRule],
    ):
        self.reference = reference
        self.sql_store = None
        self.policy = policy
        self.enable_write_queue = True

        # Async write queue and worker control
        self._queue: asyncio.Queue[_WriteItem] | None = None
        self._worker_tasks: list[asyncio.Task[Any]] = []
        self._max_write_queue_size: int = reference.max_write_queue_size
        self._num_writers: int = max(1, reference.num_writers)

    async def initialize(self):
        """Create the necessary tables if they don't exist."""
        base_store = sqlstore_impl(self.reference)
        self.sql_store = AuthorizedSqlStore(base_store, self.policy)

        # Disable write queue for SQLite since WAL mode handles concurrency
        # Keep it enabled for other backends (like Postgres) for performance
        backend_config = _SQLSTORE_BACKENDS.get(self.reference.backend)
        if backend_config and backend_config.type == StorageBackendType.SQL_SQLITE:
            self.enable_write_queue = False
            logger.debug("Write queue disabled for SQLite (WAL mode handles concurrency)")

        await self.sql_store.create_table(
            self.reference.table_name,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created": ColumnType.INTEGER,
                "model": ColumnType.STRING,
                "choices": ColumnType.JSON,
                "input_messages": ColumnType.JSON,
            },
        )

    async def shutdown(self) -> None:
        if not self._worker_tasks:
            return
        if self._queue is not None:
            await self._queue.join()
        for t in self._worker_tasks:
            if not t.done():
                t.cancel()
        for t in self._worker_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._worker_tasks.clear()

    async def flush(self) -> None:
        """Wait for all queued writes to complete. Useful for testing."""
        if self.enable_write_queue and self._queue is not None:
            await self._queue.join()

    async def _ensure_workers_started(self) -> None:
        """Ensure the async write queue workers run on the current loop."""
        if not self.enable_write_queue:
            return

        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_write_queue_size)
            logger.debug(
                "Inference store write queue created with max size and writers",
                _max_write_queue_size=self._max_write_queue_size,
                _num_writers=self._num_writers,
            )

        if not self._worker_tasks:
            for _ in range(self._num_writers):
                task = create_detached_background_task(self._worker_loop())
                self._worker_tasks.append(task)

    async def store_chat_completion(
        self, chat_completion: OpenAIChatCompletion, input_messages: list[OpenAIMessageParam]
    ) -> None:
        if self.enable_write_queue:
            await self._ensure_workers_started()
            if self._queue is None:
                raise ValueError("Inference store is not initialized")
            item = _WriteItem(chat_completion, input_messages, capture_request_context())
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.warning(
                    "Write queue full, waiting to add chat completion",
                    completion_id=getattr(chat_completion, "id", "<unknown>"),
                )
                await self._queue.put(item)
        else:
            await self._write_chat_completion(chat_completion, input_messages)

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                with activate_request_context(item.request_context):
                    await self._write_chat_completion(item.completion, item.messages)
            except Exception as e:  # noqa: BLE001
                logger.error("Error writing chat completion", error=str(e))
            finally:
                self._queue.task_done()

    async def _write_chat_completion(
        self, chat_completion: OpenAIChatCompletion, input_messages: list[OpenAIMessageParam]
    ) -> None:
        if self.sql_store is None:
            raise ValueError("Inference store is not initialized")

        data = chat_completion.model_dump()
        record_data = {
            "id": data["id"],
            "created": data["created"],
            "model": data["model"],
            "choices": data["choices"],
            "input_messages": [message.model_dump() for message in input_messages],
        }

        try:
            await self.sql_store.insert(
                table=self.reference.table_name,
                data=record_data,
            )
        except IntegrityError as e:
            # Duplicate chat completion IDs can be generated during tests especially if they are replaying
            # recorded responses across different tests. No need to warn or error under those circumstances.
            # In the wild, this is not likely to happen at all (no evidence) so we aren't really hiding any problem.

            # Check if it's a unique constraint violation
            error_message = str(e.orig) if e.orig else str(e)
            if self._is_unique_constraint_error(error_message):
                # Update the existing record instead
                await self.sql_store.update(table=self.reference.table_name, data=record_data, where={"id": data["id"]})
            else:
                # Re-raise if it's not a unique constraint error
                raise

    def _is_unique_constraint_error(self, error_message: str) -> bool:
        """Check if the error is specifically a unique constraint violation."""
        error_lower = error_message.lower()
        return any(
            indicator in error_lower
            for indicator in [
                "unique constraint failed",  # SQLite
                "duplicate key",  # PostgreSQL
                "unique violation",  # PostgreSQL alternative
                "duplicate entry",  # MySQL
            ]
        )

    async def list_chat_completions(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIChatCompletionResponse:
        """
        List chat completions from the database.

        :param after: The ID of the last chat completion to return.
        :param limit: The maximum number of chat completions to return.
        :param model: The model to filter by.
        :param order: The order to sort the chat completions by.
        """
        if not self.sql_store:
            raise ValueError("Inference store is not initialized")

        if not order:
            order = Order.desc

        where_conditions = {}
        if model:
            where_conditions["model"] = model

        paginated_result = await self.sql_store.fetch_all(
            table=self.reference.table_name,
            where=where_conditions if where_conditions else None,
            order_by=[("created", order.value)],
            cursor=("id", after) if after else None,
            limit=limit,
        )

        data = [
            OpenAICompletionWithInputMessages(
                id=row["id"],
                created=row["created"],
                model=row["model"],
                choices=row["choices"],
                input_messages=row["input_messages"],
            )
            for row in paginated_result.data
        ]
        return ListOpenAIChatCompletionResponse(
            data=data,
            has_more=paginated_result.has_more,
            first_id=data[0].id if data else "",
            last_id=data[-1].id if data else "",
        )

    async def get_chat_completion(self, completion_id: str) -> OpenAICompletionWithInputMessages:
        if not self.sql_store:
            raise ValueError("Inference store is not initialized")

        row = await self.sql_store.fetch_one(
            table=self.reference.table_name,
            where={"id": completion_id},
        )

        if not row:
            # SecureSqlStore will return None if record doesn't exist OR access is denied
            # This provides security by not revealing whether the record exists
            raise ValueError(f"Chat completion with id {completion_id} not found") from None

        return OpenAICompletionWithInputMessages(
            id=row["id"],
            created=row["created"],
            model=row["model"],
            choices=row["choices"],
            input_messages=row["input_messages"],
        )
