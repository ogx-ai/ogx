# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Reproducer for https://github.com/ogx-ai/ogx/pull/6226.
#
# ``openai_create_vector_store_file_batch`` schedules
# ``_cleanup_expired_file_batches`` with a bare ``asyncio.create_task`` whose
# result is not stored anywhere. ``shutdown()`` only walks
# ``_file_batch_tasks``, so an in-flight cleanup run is invisible to it:
#
#   1. ``await mixin.shutdown()`` returns while the cleanup is still running,
#      so shutdown is not a barrier for the work it is documented to cover
#      ("Clean up mixin resources including background tasks").
#   2. When the event loop finally goes away (server process exit), the
#      untracked task is cancelled mid-``delete`` and dies without an
#      exception or a log line, leaving expired batches half-cleaned: the
#      survivors sit in the store (and in the ``openai_file_batches``
#      registry) until the next run, which is throttled to once a day.
#
# Both tests below pin the *desired* behavior: the cleanup run is tracked and
# completes (or at least is deliberately finished) before ``shutdown()``
# returns. They fail on current main. They also fail against a fix that only
# tracks the task and *cancels* it in ``shutdown()`` (the original PR as
# pushed) — cancelling still stops the cleanup early, so the observable state
# is unchanged; the fix must await the task.
#
# The GC angle from the PR description (the task being collected mid-flight,
# RUF006) is not reproducible here: a task suspended on a loop-managed
# operation stays strongly reachable through the loop's timer/ready queue ->
# waker future -> done-callback chain, and a forced ``gc.collect()`` while the
# cleanup is mid-``delete`` does not destroy it (verified empirically on
# CPython 3.12). Keeping a strong reference is still correct, and falls out of
# the tracking the shutdown fix requires.

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock

from ogx.providers.utils.memory.openai_vector_store_mixin import (
    OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX,
    OpenAIVectorStoreMixin,
)
from ogx_api import OpenAICreateVectorStoreFileBatchRequestWithExtraBody

EXPIRED_BATCH_IDS = ["vsfb_expired_1", "vsfb_expired_2", "vsfb_expired_3"]
VECTOR_STORE_ID = "vs_1"
DELETE_DELAY = 0.05


def _expired_batch_key(batch_id: str) -> str:
    return f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch_id}"


class _FakeKVStore:
    """Deterministic in-memory KV store with a fixed per-delete latency.

    ``delete`` records each completed key and counts in-flight deletes so the
    test can observe exactly how far a cleanup run got before it was dropped.
    """

    def __init__(self, preloaded: dict[str, str]) -> None:
        self._store: dict[str, str] = dict(preloaded)
        self.deleted: list[str] = []
        self.deletes_in_flight = 0

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [v for k, v in self._store.items() if start_key <= k < end_key]

    async def delete(self, key: str) -> None:
        self.deletes_in_flight += 1
        await asyncio.sleep(DELETE_DELAY)
        self.deletes_in_flight -= 1
        self._store.pop(key, None)
        self.deleted.append(key)

    def keys(self) -> list[str]:
        return list(self._store)


class _TestMixin(OpenAIVectorStoreMixin):
    def __init__(self, kvstore: _FakeKVStore) -> None:
        super().__init__(inference_api=AsyncMock(), kvstore=kvstore)

    async def register_vector_store(self, vector_store: Any) -> None:
        pass

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        pass

    async def insert_chunks(self, request: Any) -> None:
        pass

    async def query_chunks(self, request: Any) -> Any:
        return None

    async def delete_chunks(self, request: Any) -> None:
        pass


def _make_mixin() -> tuple[_TestMixin, _FakeKVStore]:
    now = int(time.time())
    expired_entries = {
        _expired_batch_key(batch_id): json.dumps({"id": batch_id, "status": "in_progress", "expires_at": now - 1000})
        for batch_id in EXPIRED_BATCH_IDS
    }
    kv = _FakeKVStore(preloaded=expired_entries)
    mixin = _TestMixin(kvstore=kv)
    mixin.openai_vector_stores[VECTOR_STORE_ID] = {
        "id": VECTOR_STORE_ID,
        "file_ids": ["file_1"],
        "file_counts": {
            "total": 1,
            "completed": 1,
            "cancelled": 0,
            "failed": 0,
            "in_progress": 0,
        },
        "metadata": {},
    }
    # Model what initialize() would have loaded from the store (keyed by batch id).
    mixin.openai_file_batches = {
        batch_id: json.loads(expired_entries[_expired_batch_key(batch_id)]) for batch_id in EXPIRED_BATCH_IDS
    }

    # Keep the (tracked) batch-processing task in flight for the whole
    # scenario, like a batch still being processed at shutdown time.
    async def _stall(batch_id: str, batch_info: dict[str, Any], file_ids_override: list[str] | None = None) -> None:
        await asyncio.sleep(3600)

    mixin._process_file_batch_async = _stall  # type: ignore[method-assign]
    return mixin, kv


async def _wait_for_in_flight_delete(kv: _FakeKVStore) -> None:
    for _ in range(5000):
        if kv.deletes_in_flight == 1:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("cleanup task never started its first delete")


class TestExpiredBatchCleanupTask:
    async def test_shutdown_is_a_barrier_for_cleanup(self) -> None:
        """After ``shutdown()`` returns, the in-flight cleanup run must be finished."""
        mixin, kv = _make_mixin()
        batch = await mixin.openai_create_vector_store_file_batch(
            VECTOR_STORE_ID, OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_1"])
        )
        await _wait_for_in_flight_delete(kv)
        assert kv.deleted == [], "delete must still be in flight when shutdown starts"

        await mixin.shutdown()

        # Desired: shutdown waited for the cleanup, so all expired batches are gone.
        # Main: the cleanup task is untracked, so this returns immediately with
        # the first delete still in flight and nothing completed.
        assert set(kv.deleted) == {_expired_batch_key(b) for b in EXPIRED_BATCH_IDS}
        # The new batch is not expired and must survive.
        assert f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch.id}" in kv.keys()

    def test_process_exit_does_not_leave_cleanup_half_done(self) -> None:
        """When the event loop goes away after shutdown, no expired batch may be left behind."""
        mixin, kv = _make_mixin()

        batch_ids: list[str] = []

        async def scenario() -> None:
            batch = await mixin.openai_create_vector_store_file_batch(
                VECTOR_STORE_ID, OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["file_1"])
            )
            batch_ids.append(batch.id)
            await _wait_for_in_flight_delete(kv)
            await mixin.shutdown()
            # Nested asyncio.run teardown now cancels every task still on the
            # loop -- exactly what a real server shutdown does.

        asyncio.run(scenario())

        # Desired: the cleanup finished before the loop went away, leaving only
        # the new, unexpired batch.
        # Main: the untracked task is cancelled mid-delete at teardown with no
        # exception and no log line; every expired batch survives in both the
        # store and the in-memory registry.
        assert set(kv.deleted) == {_expired_batch_key(b) for b in EXPIRED_BATCH_IDS}
        assert kv.keys() == [f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}{batch_ids[0]}"]
        assert set(mixin.openai_file_batches) == {batch_ids[0]}
