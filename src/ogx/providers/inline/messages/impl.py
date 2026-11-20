# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Anthropic Messages API implementation.

Delegates to the inference API for model calls. Providers handle translation
or native passthrough via the InferenceProvider.anthropic_messages() method.

Message batch operations are implemented here as they require local state
management and do not fit the inference provider model.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ogx.core.access_control.datatypes import RouteAccessRule
from ogx.core.access_control.route_access import is_route_allowed
from ogx.core.datatypes import TenancyMode
from ogx.core.request_headers import get_authenticated_user
from ogx.core.storage.kvstore import KVStore
from ogx.core.storage.sqlstore.authorized_sqlstore import (
    AuthorizedSqlStore,
    _enhance_item_with_access_control,
)
from ogx.log import get_logger
from ogx_api import Inference, RouteAccessDeniedError
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType
from ogx_api.messages import Messages
from ogx_api.messages.models import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
    CancelMessageBatchRequest,
    CreateMessageBatchRequest,
    ListMessageBatchesRequest,
    ListMessageBatchesResponse,
    MessageBatch,
    MessageBatchCanceledResult,
    MessageBatchErroredResult,
    MessageBatchIndividualResponse,
    MessageBatchRequestCounts,
    MessageBatchSucceededResult,
    RetrieveMessageBatchRequest,
    RetrieveMessageBatchResultsRequest,
    _AnthropicErrorDetail,
)

from .config import MessagesConfig

_TABLE_MESSAGE_BATCHES = "message_batches"
# Legacy key prefix from the pre-authorization implementation, where batch
# metadata lived in the KV store with no owner information. Retained so the
# one-way startup migration can find and relocate those rows.
_BATCH_PREFIX = "msgbatch:"
# The endpoint every batch request is executed against; batch processing calls the
# inference API directly in-process and never passes through
# RouteAuthorizationMiddleware, so this endpoint's route_policy is re-checked at
# creation time.
_MESSAGES_ENDPOINT = "/v1/messages"
_BATCH_RESULTS_PREFIX = "msgbatch_results:"
_BATCH_EXPIRY_HOURS = 24

logger = get_logger(name=__name__, category="messages")


@dataclass
class _BatchContext:
    """Internal bundle of a batch id with its original creation request."""

    batch_id: str
    request: CreateMessageBatchRequest


class BuiltinMessagesImpl(Messages):
    """Anthropic Messages API adapter that delegates to the inference API."""

    def __init__(
        self,
        config: MessagesConfig,
        inference_api: Inference,
        kvstore: KVStore,
        sql_store: AuthorizedSqlStore,
        route_policy: list[RouteAccessRule] | None = None,
    ):
        self.config = config
        self.inference_api = inference_api
        self.kvstore = kvstore
        self.sql_store = sql_store
        self.route_policy = route_policy or []
        self._processing_tasks: dict[str, asyncio.Task] = {}
        self._batch_semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        self._update_lock = asyncio.Lock()
        # Partial results held in memory so a cancellation can finalize with the
        # truly-completed outcomes, not "all canceled". Keyed by batch_id.
        self._partial_results: dict[str, list[dict[str, Any]]] = {}

        # this is to allow tests to disable background processing
        self.process_batches = True

    async def initialize(self) -> None:
        await self.sql_store.create_table(
            _TABLE_MESSAGE_BATCHES,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.STRING,
                "batch_data": ColumnType.JSON,
            },
        )
        await self._migrate_legacy_kv_batches()

    async def _migrate_legacy_kv_batches(self) -> None:
        """Relocate batch metadata written by the pre-authorization implementation.

        Before batches moved to the SQL store, their metadata lived in the KV store
        under unowned keys, so existing deployments have rows the new storage path
        cannot see. Re-inserting them on startup keeps previously created batches
        readable after an upgrade. The legacy rows carry no owner information;
        inserting them without a request context marks them as server-owned rows
        (owner_principal=""), which under the default policy preserves exactly the
        pre-upgrade visibility. The per-row existence check keeps the migration a
        no-op on later startups, and the insert is conflict-tolerant so several
        server processes sharing the same storage (e.g. a multi-replica deployment)
        can run it concurrently without racing on the primary key. The legacy KV
        entries are left in place so the data remains recoverable.
        """
        legacy_values = await self.kvstore.values_in_range(f"{_BATCH_PREFIX}", f"{_BATCH_PREFIX}\xff")
        migrated = 0
        for value in legacy_values:
            try:
                batch = MessageBatch.model_validate_json(value)
            except Exception:
                logger.warning("Skipping unparseable legacy message batch entry during migration")
                continue
            # Bootstrap-time check: the migration runs outside any request context, so
            # it must see rows of every tenant to stay a no-op in multi-tenant mode.
            existing = await self.sql_store.sql_store.fetch_all(
                table=_TABLE_MESSAGE_BATCHES, where={"id": batch.id}, limit=1
            )
            if existing.data:
                continue
            row = _enhance_item_with_access_control(
                {
                    "id": batch.id,
                    "created_at": batch.created_at,
                    "batch_data": batch.model_dump(),
                },
                None,
                self.sql_store.tenancy_mode,
                self.sql_store.default_tenant_id,
            )
            await self.sql_store.sql_store.insert_do_nothing(table=_TABLE_MESSAGE_BATCHES, data=row)
            migrated += 1
        if migrated:
            if self.sql_store.tenancy_mode == TenancyMode.MULTI:
                # In multi-tenant mode there is no tenant to attribute the legacy rows
                # to; they are kept in the table but invisible to tenant-scoped users.
                logger.warning(
                    "Migrated legacy message batches without owner or tenant information; "
                    "they are not visible to tenant-scoped users in multi-tenant mode",
                    migrated=migrated,
                )
            else:
                logger.info("Migrated legacy message batches from KV store", migrated=migrated)

    async def shutdown(self) -> None:
        if self._processing_tasks:
            logger.info(
                "Shutdown initiated with active batch processing tasks",
                active_tasks=len(self._processing_tasks),
            )

    async def create_message(
        self,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        return await self.inference_api.anthropic_messages(request)

    async def count_message_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        return await self.inference_api.anthropic_count_tokens(request)

    # -- Message Batches --

    # SECURITY NOTE: batch processing runs in a background asyncio.Task spawned from
    # create_message_batch() while still inside the creating request's context.
    # asyncio.create_task() snapshots contextvars at spawn time, so that task retains
    # the creating user's authenticated identity (get_authenticated_user()) and provider
    # data (x-ogx-provider-data) for its entire lifetime -- per-request inference
    # credentials and AuthorizedSqlStore-backed policy checks (list_message_batches,
    # retrieve_message_batch, cancel_message_batch, retrieve_message_batch_results) are
    # therefore enforced the same way they are for a live, synchronous request.
    #
    # This guarantee holds only within a single server process's lifetime: today there is
    # no code path that processes a batch outside its creating request's context. If
    # restart resumption is ever implemented, it must explicitly re-enter the original
    # user's RequestProviderDataContext before resuming -- resuming with no context would
    # silently fall back to server-configured credentials and bypass per-user policy.
    #
    # The above is resource-level (AuthorizedSqlStore) and per-request-credential
    # enforcement; it does NOT cover route-level authorization. RouteAuthorizationMiddleware
    # only inspects the outer HTTP request's own path (e.g. POST /v1/messages/batches), so
    # a route_policy rule forbidding a user from /v1/messages would not stop that user from
    # creating a batch: batch processing calls the inference API directly in-process, never
    # through that middleware. The is_route_allowed() check in create_message_batch
    # re-applies the same route_policy to the batch endpoint at creation time so that
    # guarantee holds for message batches too.

    async def create_message_batch(
        self,
        request: CreateMessageBatchRequest,
    ) -> MessageBatch:
        seen_ids: set[str] = set()
        for req in request.requests:
            if req.custom_id in seen_ids:
                raise ValueError(f"Failed to create batch: duplicate custom_id '{req.custom_id}'")
            seen_ids.add(req.custom_id)

        # Every request line in the batch is executed against the Messages endpoint
        # in-process, so checking that endpoint once here is equivalent to checking every
        # line. This runs the same route_policy evaluation RouteAuthorizationMiddleware
        # applies to a live request to that endpoint.
        if not is_route_allowed(_MESSAGES_ENDPOINT, get_authenticated_user(), self.route_policy):
            raise RouteAccessDeniedError(_MESSAGES_ENDPOINT)

        now = datetime.now(UTC)
        batch_id = f"msgbatch_{uuid.uuid4().hex[:24]}"
        batch = MessageBatch(
            id=batch_id,
            processing_status="in_progress",
            request_counts=MessageBatchRequestCounts(processing=len(request.requests)),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=_BATCH_EXPIRY_HOURS)).isoformat(),
        )

        await self.sql_store.insert(
            table=_TABLE_MESSAGE_BATCHES,
            data={
                "id": batch_id,
                "created_at": now.isoformat(),
                "batch_data": batch.model_dump(),
            },
        )
        logger.info("Created message batch", batch_id=batch_id, request_count=len(request.requests))

        ctx = _BatchContext(batch_id=batch_id, request=request)
        self._partial_results[batch_id] = []
        if self.process_batches:
            task = asyncio.create_task(self._process_message_batch(ctx))
            self._processing_tasks[batch_id] = task
        return batch

    async def _load_batch(self, batch_id: str) -> MessageBatch:
        record = await self.sql_store.fetch_one(table=_TABLE_MESSAGE_BATCHES, where={"id": batch_id})
        if record is None:
            raise KeyError(batch_id)
        return MessageBatch.model_validate(record["batch_data"])

    async def _save_batch(self, batch: MessageBatch) -> None:
        await self.sql_store.update(
            table=_TABLE_MESSAGE_BATCHES,
            data={"batch_data": batch.model_dump()},
            where={"id": batch.id},
        )

    async def retrieve_message_batch(self, request: RetrieveMessageBatchRequest) -> MessageBatch:
        return await self._load_batch(request.batch_id)

    async def list_message_batches(
        self,
        request: ListMessageBatchesRequest,
    ) -> ListMessageBatchesResponse:
        # Filtering is delegated to AuthorizedSqlStore.fetch_all, which enforces the
        # configured access policy at the row level -- this returns only batches the caller
        # owns (or is otherwise permitted to see under that policy), not all batches in the
        # table.
        results = await self.sql_store.fetch_all(table=_TABLE_MESSAGE_BATCHES)

        batches = [MessageBatch.model_validate(row["batch_data"]) for row in results.data]
        batches.sort(key=lambda b: b.created_at, reverse=True)

        if request.after_id:
            idx = next((i for i, b in enumerate(batches) if b.id == request.after_id), None)
            if idx is not None:
                batches = batches[idx + 1 :]

        if request.before_id:
            idx = next((i for i, b in enumerate(batches) if b.id == request.before_id), None)
            if idx is not None:
                batches = batches[:idx]

        has_more = len(batches) > request.limit
        batches = batches[: request.limit]

        return ListMessageBatchesResponse(
            data=batches,
            has_more=has_more,
            first_id=batches[0].id if batches else None,
            last_id=batches[-1].id if batches else None,
        )

    async def cancel_message_batch(self, request: CancelMessageBatchRequest) -> MessageBatch:
        batch_id = request.batch_id
        # Acquire the lock before reading so we eliminate the read-check-write
        # race with concurrent cancel/finalize calls. The task is cancelled
        # only after the lock is released to avoid awaiting cancellation while
        # holding the lock (the cancelled task itself needs the lock to finalize).
        task_to_cancel: asyncio.Task | None = None
        async with self._update_lock:
            batch = await self._load_batch(batch_id)
            if batch.processing_status == "ended":
                raise ValueError(f"Failed to cancel batch '{batch_id}': batch has already ended")
            if batch.processing_status != "canceling":
                batch.processing_status = "canceling"
                batch.cancel_initiated_at = datetime.now(UTC).isoformat()
                await self._save_batch(batch)
                task_to_cancel = self._processing_tasks.get(batch_id)

        if task_to_cancel is not None:
            task_to_cancel.cancel()

        return await self._load_batch(batch_id)

    async def retrieve_message_batch_results(
        self,
        request: RetrieveMessageBatchResultsRequest,
    ) -> AsyncIterator[MessageBatchIndividualResponse]:
        batch_id = request.batch_id
        batch = await self._load_batch(batch_id)
        if batch.processing_status != "ended":
            raise ValueError(
                f"Failed to retrieve batch results for '{batch_id}': batch has not finished processing (status: {batch.processing_status})"
            )

        data = await self.kvstore.get(f"{_BATCH_RESULTS_PREFIX}{batch_id}")
        if data is None:

            async def _empty_iter() -> AsyncIterator[MessageBatchIndividualResponse]:
                return
                yield  # pragma: no cover — makes this an async generator

            return _empty_iter()

        parsed: list[dict[str, Any]] = json.loads(data)
        results_list: list[MessageBatchIndividualResponse] = [
            MessageBatchIndividualResponse.model_validate(item) for item in parsed
        ]

        async def _iter_results(
            items: list[MessageBatchIndividualResponse],
        ) -> AsyncIterator[MessageBatchIndividualResponse]:
            for item in items:
                yield item

        return _iter_results(results_list)

    async def _process_message_batch(self, ctx: _BatchContext) -> None:
        try:
            async with self._batch_semaphore:
                await self._process_message_batch_impl(ctx)
        except asyncio.CancelledError:
            await self._finalize_batch_canceled(ctx)
        except Exception:
            logger.exception("Failed to process message batch", batch_id=ctx.batch_id)
            await self._finalize_batch_error(ctx)
        finally:
            self._processing_tasks.pop(ctx.batch_id, None)
            self._partial_results.pop(ctx.batch_id, None)

    async def _process_message_batch_impl(self, ctx: _BatchContext) -> None:
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests_per_batch)
        partial = self._partial_results[ctx.batch_id]

        async def process_one(custom_id: str, params: AnthropicCreateMessageRequest) -> None:
            async with semaphore:
                # Force non-streaming for batch requests
                params.stream = False
                try:
                    response = await self.create_message(params)
                    if isinstance(response, AnthropicMessageResponse):
                        result_obj = MessageBatchIndividualResponse(
                            custom_id=custom_id,
                            result=MessageBatchSucceededResult(message=response),
                        )
                    else:
                        result_obj = MessageBatchIndividualResponse(
                            custom_id=custom_id,
                            result=MessageBatchErroredResult(
                                error=_AnthropicErrorDetail(type="api_error", message="Unexpected streaming response"),
                            ),
                        )
                except Exception as e:
                    result_obj = MessageBatchIndividualResponse(
                        custom_id=custom_id,
                        result=MessageBatchErroredResult(
                            error=_AnthropicErrorDetail(type="api_error", message=str(e)),
                        ),
                    )
                # Append to the shared partial-results list. Single-threaded
                # asyncio means the append itself is safe; cancellation that
                # arrives after this point still sees the completed entry.
                partial.append(result_obj.model_dump())

        tasks = [process_one(req.custom_id, req.params) for req in ctx.request.requests]
        await asyncio.gather(*tasks)

        succeeded = sum(1 for r in partial if r["result"]["type"] == "succeeded")
        errored = len(partial) - succeeded

        await self.kvstore.set(f"{_BATCH_RESULTS_PREFIX}{ctx.batch_id}", json.dumps(partial))

        async with self._update_lock:
            batch = await self._load_batch(ctx.batch_id)
            batch.processing_status = "ended"
            batch.ended_at = datetime.now(UTC).isoformat()
            batch.request_counts = MessageBatchRequestCounts(
                processing=0,
                succeeded=succeeded,
                errored=errored,
            )
            batch.results_url = f"/v1/messages/batches/{ctx.batch_id}/results"
            await self._save_batch(batch)

        logger.info(
            "Message batch completed",
            batch_id=ctx.batch_id,
            succeeded=succeeded,
            errored=errored,
        )

    async def _finalize_batch_canceled(self, ctx: _BatchContext) -> None:
        # Source of truth for completed work is the in-memory partial list,
        # which is updated as each request finishes. This avoids the race
        # where _process_message_batch_impl had not yet flushed to kvstore.
        results: list[dict[str, Any]] = list(self._partial_results.get(ctx.batch_id, []))
        completed_ids = {r["custom_id"] for r in results}
        succeeded = sum(1 for r in results if r["result"]["type"] == "succeeded")
        errored = len(results) - succeeded
        canceled = 0

        for req in ctx.request.requests:
            if req.custom_id not in completed_ids:
                results.append(
                    MessageBatchIndividualResponse(
                        custom_id=req.custom_id,
                        result=MessageBatchCanceledResult(),
                    ).model_dump()
                )
                canceled += 1

        await self.kvstore.set(f"{_BATCH_RESULTS_PREFIX}{ctx.batch_id}", json.dumps(results))

        async with self._update_lock:
            batch = await self._load_batch(ctx.batch_id)
            batch.processing_status = "ended"
            batch.ended_at = datetime.now(UTC).isoformat()
            batch.request_counts = MessageBatchRequestCounts(
                processing=0,
                succeeded=succeeded,
                errored=errored,
                canceled=canceled,
            )
            batch.results_url = f"/v1/messages/batches/{ctx.batch_id}/results"
            await self._save_batch(batch)

        logger.info(
            "Message batch canceled",
            batch_id=ctx.batch_id,
            succeeded=succeeded,
            errored=errored,
            canceled=canceled,
        )

    async def _finalize_batch_error(self, ctx: _BatchContext) -> None:
        # Preserve any partial results that completed before the error
        results: list[dict[str, Any]] = list(self._partial_results.get(ctx.batch_id, []))
        completed_ids = {r["custom_id"] for r in results}
        succeeded = sum(1 for r in results if r["result"]["type"] == "succeeded")
        errored = len(results) - succeeded

        for req in ctx.request.requests:
            if req.custom_id not in completed_ids:
                results.append(
                    MessageBatchIndividualResponse(
                        custom_id=req.custom_id,
                        result=MessageBatchErroredResult(
                            error=_AnthropicErrorDetail(type="api_error", message="Batch processing failed"),
                        ),
                    ).model_dump()
                )
                errored += 1

        await self.kvstore.set(f"{_BATCH_RESULTS_PREFIX}{ctx.batch_id}", json.dumps(results))

        async with self._update_lock:
            batch = await self._load_batch(ctx.batch_id)
            batch.processing_status = "ended"
            batch.ended_at = datetime.now(UTC).isoformat()
            batch.request_counts = MessageBatchRequestCounts(
                processing=0,
                succeeded=succeeded,
                errored=errored,
            )
            batch.results_url = f"/v1/messages/batches/{ctx.batch_id}/results"
            await self._save_batch(batch)

        logger.info(
            "Message batch errored",
            batch_id=ctx.batch_id,
            succeeded=succeeded,
            errored=errored,
        )
