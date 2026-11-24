# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for BuiltinMessagesImpl — batch operations and inference delegation.

Authorization tests mirror the reference batches provider's per-user scoping
tests (see tests/unit/providers/batches/test_reference.py): message batch
metadata lives in an AuthorizedSqlStore table, so list/retrieve/cancel/results
are scoped to the creating user.
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import ogx.core.storage.sqlstore.sqlstore as sqlstore_module
from ogx.core.access_control.datatypes import RouteAccessRule, RouteScope
from ogx.core.datatypes import TenancyMode, User
from ogx.core.request_headers import RequestProviderDataContext, get_authenticated_user
from ogx.core.storage.datatypes import KVStoreReference, SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import (
    _sqlstore_impl,
    register_sqlstore_backends,
    shutdown_sqlstore_backends,
)
from ogx.providers.inline.messages.config import MessagesConfig
from ogx.providers.inline.messages.impl import BuiltinMessagesImpl
from ogx_api import RouteAccessDeniedError
from ogx_api.messages.models import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessage,
    AnthropicMessageResponse,
    AnthropicUsage,
    CancelMessageBatchRequest,
    CreateMessageBatchRequest,
    ListMessageBatchesRequest,
    MessageBatch,
    MessageBatchRequestCounts,
    MessageBatchRequestParams,
    RetrieveMessageBatchRequest,
    RetrieveMessageBatchResultsRequest,
)


@pytest.fixture
async def impl():
    mock_inference = AsyncMock()
    mock_kvstore = AsyncMock()
    mock_kvstore.values_in_range = AsyncMock(return_value=[])
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_message_batches.db"
        backend_name = "sql_msgbatches_test"
        register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=str(db_path))})
        config = MessagesConfig(
            sqlstore=SqlStoreReference(backend=backend_name, table_name="message_batches"),
            kvstore=KVStoreReference(backend="kv_default", namespace="test"),
        )

        # Create sql_store and mock APIs
        base_sql_store = await _sqlstore_impl(config.sqlstore)
        sql_store = AuthorizedSqlStore(base_sql_store, policy=[])

        impl = BuiltinMessagesImpl(
            config=config,
            inference_api=mock_inference,
            kvstore=mock_kvstore,
            sql_store=sql_store,
        )
        await impl.initialize()

        # unit tests should not require background processing
        impl.process_batches = False

        yield impl

        await impl.shutdown()
        # Dispose the engine while the tempdir still exists; leaving pooled
        # connections open over a deleted sqlite file makes later tests flaky.
        await shutdown_sqlstore_backends()


class TestCreateMessageDelegation:
    async def test_create_message_delegates_to_inference_api(self, impl):
        request = AnthropicCreateMessageRequest(
            model="claude-sonnet-4-20250514",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        expected_response = AnthropicMessageResponse(
            id="msg_abc",
            content=[],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )
        impl.inference_api.anthropic_messages.return_value = expected_response

        result = await impl.create_message(request)

        impl.inference_api.anthropic_messages.assert_awaited_once_with(request)
        assert result is expected_response


class TestCountTokensDelegation:
    async def test_count_message_tokens_delegates_to_inference_api(self, impl):
        request = AnthropicCountTokensRequest(
            model="claude-sonnet-4-20250514",
            system="You are helpful.",
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        expected_response = AnthropicCountTokensResponse(input_tokens=10)
        impl.inference_api.anthropic_count_tokens.return_value = expected_response

        result = await impl.count_message_tokens(request)

        impl.inference_api.anthropic_count_tokens.assert_awaited_once_with(request)
        assert result is expected_response


class TestMessageBatchCreation:
    @staticmethod
    def _batch_request(custom_id="req-1"):
        return CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id=custom_id,
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ]
        )

    async def test_create_batch_detects_duplicate_custom_ids(self, impl):
        request = CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id="dup-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
                MessageBatchRequestParams(
                    custom_id="dup-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ],
        )

        with pytest.raises(ValueError, match="duplicate custom_id"):
            await impl.create_message_batch(request)

    async def test_create_batch_returns_in_progress_status(self, impl):
        batch = await impl.create_message_batch(self._batch_request())

        assert batch.id.startswith("msgbatch_")
        assert batch.processing_status == "in_progress"
        assert batch.request_counts.processing == 1

    async def test_cancel_already_ended_batch_raises(self, impl):
        batch = await impl.create_message_batch(self._batch_request())
        batch_id = batch.id

        ended_batch = MessageBatch(
            id=batch_id,
            processing_status="ended",
            request_counts=batch.request_counts,
            created_at=batch.created_at,
            expires_at=batch.expires_at,
            ended_at=datetime.now(UTC).isoformat(),
        )
        await impl._save_batch(ended_batch)

        with pytest.raises(ValueError, match="batch has already ended"):
            await impl.cancel_message_batch(CancelMessageBatchRequest(batch_id=batch_id))

    async def test_retrieve_results_not_ended_raises(self, impl):
        batch = await impl.create_message_batch(self._batch_request())

        # Background processing is disabled in the fixture, so the batch is still in_progress.
        with pytest.raises(ValueError, match="batch has not finished processing"):
            results_iter = await impl.retrieve_message_batch_results(
                RetrieveMessageBatchResultsRequest(batch_id=batch.id)
            )
            await results_iter.__anext__()


class TestLegacyKvMigration:
    """Startup migration of batch metadata written by the pre-authorization
    implementation (KV store keys with no owner information)."""

    @staticmethod
    def _legacy_batch_json(batch_id="msgbatch_legacy123"):
        now = datetime.now(UTC)
        return MessageBatch(
            id=batch_id,
            processing_status="in_progress",
            request_counts=MessageBatchRequestCounts(processing=1),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=24)).isoformat(),
        ).model_dump_json()

    async def test_migrates_legacy_kv_batches(self, impl):
        """Legacy KV entries are re-inserted into the SQL table on startup, keeping
        previously created batches readable after an upgrade."""
        impl.kvstore.values_in_range = AsyncMock(return_value=[self._legacy_batch_json()])

        await impl._migrate_legacy_kv_batches()

        record = await impl.sql_store.fetch_one(table="message_batches", where={"id": "msgbatch_legacy123"})
        assert record is not None
        batch = MessageBatch.model_validate(record["batch_data"])
        assert batch.processing_status == "in_progress"
        assert batch.request_counts.processing == 1

    async def test_migration_is_idempotent(self, impl):
        """Restarting the server does not duplicate already-migrated rows."""
        impl.kvstore.values_in_range = AsyncMock(return_value=[self._legacy_batch_json()])

        await impl._migrate_legacy_kv_batches()
        await impl._migrate_legacy_kv_batches()

        rows = await impl.sql_store.sql_store.fetch_all(table="message_batches")
        assert len(rows.data) == 1

    async def test_migration_is_safe_under_concurrent_startup(self, impl):
        """Several server processes sharing the same storage (e.g. a multi-replica
        deployment) can run the migration concurrently; the conflict-tolerant insert
        absorbs the primary-key race instead of failing startup."""
        impl.kvstore.values_in_range = AsyncMock(return_value=[self._legacy_batch_json()])

        # Warm the store first (single sequential operation). This mirrors the
        # multi-replica precondition — the table already exists because an earlier
        # startup created it — and isolates the check-then-insert interleaving
        # under test from the store's lazy first-use engine initialization.
        await impl.sql_store.sql_store.fetch_all(table="message_batches")

        await asyncio.gather(impl._migrate_legacy_kv_batches(), impl._migrate_legacy_kv_batches())

        rows = await impl.sql_store.sql_store.fetch_all(table="message_batches")
        assert len(rows.data) == 1

        # Exercise the conflict path directly (independent of the interleaving above):
        # re-inserting an existing row must be a silent no-op, not an error.
        await impl.sql_store.sql_store.insert_do_nothing(table="message_batches", data=rows.data[0])
        rows = await impl.sql_store.sql_store.fetch_all(table="message_batches")
        assert len(rows.data) == 1

    async def test_migration_worst_case_both_pass_existence_check_before_either_inserts(self, impl):
        """Force the worst-case multi-replica interleaving deterministically: both
        'replicas' pass the existence check before either inserts. Only the
        conflict-tolerant insert keeps the result correct (a plain insert here would
        raise a primary-key violation and crash server startup)."""
        impl.kvstore.values_in_range = AsyncMock(return_value=[self._legacy_batch_json()])

        base_store = impl.sql_store.sql_store
        # Warm the store first so the barrier below gates the check-then-insert
        # race, not the store's lazy first-use engine initialization.
        await base_store.fetch_all(table="message_batches")
        real_fetch_all = base_store.fetch_all
        arrivals = 0
        release = asyncio.Event()

        async def gated_fetch_all(*args, **kwargs):
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                release.set()
            await release.wait()
            return await real_fetch_all(*args, **kwargs)

        base_store.fetch_all = gated_fetch_all
        try:
            await asyncio.gather(impl._migrate_legacy_kv_batches(), impl._migrate_legacy_kv_batches())
        finally:
            base_store.fetch_all = real_fetch_all

        assert arrivals == 2
        rows = await impl.sql_store.sql_store.fetch_all(table="message_batches")
        assert len(rows.data) == 1

    async def test_migration_skips_unparseable_entries(self, impl):
        """A corrupt legacy entry is skipped without blocking startup."""
        impl.kvstore.values_in_range = AsyncMock(return_value=["{not json", self._legacy_batch_json()])

        await impl._migrate_legacy_kv_batches()

        rows = await impl.sql_store.sql_store.fetch_all(table="message_batches")
        assert [row["id"] for row in rows.data] == ["msgbatch_legacy123"]

    async def test_migration_in_multi_tenant_mode_leaves_rows_invisible_to_tenants(self):
        """In multi-tenant mode there is no tenant to attribute legacy rows to, so
        they are migrated as tenantless rows: present in the table, but not visible
        to tenant-scoped users (fail-closed)."""
        legacy_batch_id = "msgbatch_legacy123"
        now = datetime.now(UTC)
        legacy_json = MessageBatch(
            id=legacy_batch_id,
            processing_status="in_progress",
            request_counts=MessageBatchRequestCounts(processing=1),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=24)).isoformat(),
        ).model_dump_json()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_message_batches_multi.db"
            backend_name = "sql_msgbatches_multi_test"
            register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=str(db_path))})
            reference = SqlStoreReference(backend=backend_name, table_name="message_batches")
            base_sql_store = await _sqlstore_impl(reference)
            sql_store = AuthorizedSqlStore(base_sql_store, policy=[], tenancy_mode=TenancyMode.MULTI)
            mock_kvstore = AsyncMock()
            mock_kvstore.values_in_range = AsyncMock(return_value=[legacy_json])
            impl = BuiltinMessagesImpl(
                config=MessagesConfig(
                    sqlstore=reference,
                    kvstore=KVStoreReference(backend="kv_default", namespace="test"),
                ),
                inference_api=AsyncMock(),
                kvstore=mock_kvstore,
                sql_store=sql_store,
            )

            await impl.initialize()

            rows = await sql_store.sql_store.fetch_all(table="message_batches")
            assert len(rows.data) == 1
            assert rows.data[0]["tenant_id"] == ""

            tenant_user = User(principal="alice", attributes={}, tenant_id="tenant-a")
            with RequestProviderDataContext(user=tenant_user):
                with pytest.raises(KeyError):
                    await impl._load_batch(legacy_batch_id)


class TestMessageBatchAuthorization:
    """Per-user scoping of message batches and credential propagation into
    background processing (mirrors the reference batches provider's tests)."""

    @staticmethod
    def _batch_request(custom_id="req-1"):
        return CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id=custom_id,
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ]
        )

    async def test_list_message_batches_scoped_to_creating_user(self, impl):
        """list_message_batches should only return batches owned by the authenticated
        caller, not every batch in the store (see AuthorizedSqlStore.fetch_all)."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            alice_batch = await impl.create_message_batch(self._batch_request("req-alice"))

        with RequestProviderDataContext(user=bob):
            bob_batch = await impl.create_message_batch(self._batch_request("req-bob"))

        with RequestProviderDataContext(user=alice):
            alice_list = await impl.list_message_batches(ListMessageBatchesRequest())
        alice_ids = {b.id for b in alice_list.data}
        assert alice_batch.id in alice_ids
        assert bob_batch.id not in alice_ids

        with RequestProviderDataContext(user=bob):
            bob_list = await impl.list_message_batches(ListMessageBatchesRequest())
        bob_ids = {b.id for b in bob_list.data}
        assert bob_batch.id in bob_ids
        assert alice_batch.id not in bob_ids

    async def test_retrieve_message_batch_denied_for_other_user(self, impl):
        """A user cannot retrieve a batch created by a different user."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            alice_batch = await impl.create_message_batch(self._batch_request("req-alice"))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(KeyError):
                await impl.retrieve_message_batch(RetrieveMessageBatchRequest(batch_id=alice_batch.id))

        with RequestProviderDataContext(user=alice):
            retrieved = await impl.retrieve_message_batch(RetrieveMessageBatchRequest(batch_id=alice_batch.id))
        assert retrieved.id == alice_batch.id

    async def test_cancel_message_batch_denied_for_other_user(self, impl):
        """A user cannot cancel a batch created by a different user."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            alice_batch = await impl.create_message_batch(self._batch_request("req-alice"))

        with RequestProviderDataContext(user=bob):
            with pytest.raises(KeyError):
                await impl.cancel_message_batch(CancelMessageBatchRequest(batch_id=alice_batch.id))

    async def test_retrieve_message_batch_results_denied_for_other_user(self, impl):
        """A user cannot stream the results of a batch created by a different user,
        even after it has ended."""
        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        bob = User(principal="bob", attributes={"roles": ["bob-role"]})

        with RequestProviderDataContext(user=alice):
            alice_batch = await impl.create_message_batch(self._batch_request("req-alice"))
            await impl._save_batch(
                MessageBatch(
                    id=alice_batch.id,
                    processing_status="ended",
                    request_counts=alice_batch.request_counts,
                    created_at=alice_batch.created_at,
                    expires_at=alice_batch.expires_at,
                    ended_at=datetime.now(UTC).isoformat(),
                )
            )

        with RequestProviderDataContext(user=bob):
            with pytest.raises(KeyError):
                await impl.retrieve_message_batch_results(RetrieveMessageBatchResultsRequest(batch_id=alice_batch.id))

    async def test_batch_processing_preserves_creating_users_context(self, impl):
        """The background task spawned by create_message_batch retains the creating
        user's authenticated identity and provider data for its entire lifetime, since
        asyncio.create_task() snapshots contextvars at spawn time. This is what lets
        per-request inference credentials (x-ogx-provider-data) and policy checks apply to
        batch-processed requests the same way they do for a live request."""
        impl.process_batches = True

        captured_user = None
        captured_event = asyncio.Event()

        async def fake_process_message_batch_impl(ctx):
            nonlocal captured_user
            captured_user = get_authenticated_user()
            captured_event.set()

        impl._process_message_batch_impl = fake_process_message_batch_impl

        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        with RequestProviderDataContext(user=alice):
            await impl.create_message_batch(self._batch_request())

        await asyncio.wait_for(captured_event.wait(), timeout=1)

        assert captured_user is not None
        assert captured_user.principal == "alice"

    async def test_batch_processing_completes_with_scoped_row_updates(self, impl):
        """The real background-processing path -- only the terminal inference call
        is mocked -- must load and update the SQL row under the creating user's
        snapshotted context, and the completed batch must remain scoped to that
        user."""
        impl.process_batches = True
        impl.inference_api.anthropic_messages = AsyncMock(
            return_value=AnthropicMessageResponse(
                id="msg_abc",
                content=[],
                model="m",
                stop_reason="end_turn",
                usage=AnthropicUsage(input_tokens=10, output_tokens=5),
            )
        )

        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        with RequestProviderDataContext(user=alice):
            batch = await impl.create_message_batch(self._batch_request())
            task = impl._processing_tasks[batch.id]

        await asyncio.wait_for(task, timeout=1)

        with RequestProviderDataContext(user=alice):
            retrieved = await impl.retrieve_message_batch(RetrieveMessageBatchRequest(batch_id=batch.id))
        assert retrieved.processing_status == "ended"
        assert retrieved.request_counts.succeeded == 1
        impl.inference_api.anthropic_messages.assert_awaited_once()

        bob = User(principal="bob", attributes={"roles": ["bob-role"]})
        with RequestProviderDataContext(user=bob):
            with pytest.raises(KeyError):
                await impl.retrieve_message_batch(RetrieveMessageBatchRequest(batch_id=batch.id))

    async def test_create_message_batch_denied_by_route_policy(self, impl):
        """create_message_batch must enforce the same route_policy that
        RouteAuthorizationMiddleware applies to a live HTTP request. A route forbidden
        there must also be denied here, since background processing calls the
        inference API directly in-process and never passes through that middleware.

        This calls provider.route_policy through is_route_allowed() -- the exact
        function RouteAuthorizationMiddleware itself calls (see
        ogx.core.access_control.route_access) -- so it exercises the identical
        evaluation a live request to /v1/messages would get, not a separate
        reimplementation that could silently drift from the real one."""
        impl.route_policy = [
            RouteAccessRule(forbid=RouteScope(paths="/v1/messages"), description="No messages for this caller"),
            RouteAccessRule(permit=RouteScope(paths="*"), description="Allow everything else"),
        ]

        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        with RequestProviderDataContext(user=alice):
            with pytest.raises(RouteAccessDeniedError):
                await impl.create_message_batch(self._batch_request())

        # Denial happens at creation: no row was persisted and background
        # processing never had a chance to run the forbidden request.
        with RequestProviderDataContext(user=alice):
            listed = await impl.list_message_batches(ListMessageBatchesRequest())
        assert listed.data == []
        impl.inference_api.anthropic_messages.assert_not_called()


class TestProviderFactory:
    async def test_unresolvable_sql_backend_error_names_provider_and_fix(self):
        """A hand-written config whose storage section defines no SQL backend
        must fail startup with a message that names the messages provider and
        the remedy, not a bare backend-resolution error."""
        from ogx.providers.inline.messages import get_provider_impl

        saved_backends = dict(sqlstore_module._SQLSTORE_BACKENDS)
        try:
            sqlstore_module._SQLSTORE_BACKENDS.clear()
            config = MessagesConfig.model_validate(
                {"kvstore": {"namespace": "message_batches", "backend": "kv_default"}}
            )
            with pytest.raises(ValueError, match="Failed to initialize the messages provider"):
                await get_provider_impl(config, {}, [], [])
        finally:
            sqlstore_module._SQLSTORE_BACKENDS.clear()
            sqlstore_module._SQLSTORE_BACKENDS.update(saved_backends)
