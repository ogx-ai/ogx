# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the durable job queue."""

import time

import pytest

from ogx.core.datatypes import User
from ogx.core.jobs.queue import JobQueue
from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx_api.common.job_types import JobStatus


async def _enqueue(queue: JobQueue, **overrides):
    payload = overrides.pop("payload", {"request": {"file_id": "f1"}})
    return await queue.enqueue(
        api=overrides.pop("api", "file_processors"),
        provider_id=overrides.pop("provider_id", "p1"),
        method=overrides.pop("method", "process_file"),
        payload=payload,
        **overrides,
    )


async def test_enqueue_starts_scheduled(queue: JobQueue):
    record = await _enqueue(queue)
    assert record.status == JobStatus.scheduled
    assert record.attempts == 0
    fetched = await queue.get(record.job_id)
    assert fetched is not None
    assert fetched.payload == {"request": {"file_id": "f1"}}


async def test_lease_claims_one_job_exclusively(queue: JobQueue):
    record = await _enqueue(queue)

    leased = await queue.lease("worker-A")
    assert leased is not None
    assert leased.job_id == record.job_id
    assert leased.status == JobStatus.in_progress
    assert leased.lease_owner == "worker-A"
    assert leased.attempts == 1

    # No other runnable job exists, so a second worker gets nothing.
    assert await queue.lease("worker-B") is None


async def test_complete_records_result(queue: JobQueue):
    record = await _enqueue(queue)
    leased = await queue.lease("worker-A")
    await queue.complete(leased.job_id, "worker-A", {"chunks": [], "metadata": {}})

    done = await queue.get(record.job_id)
    assert done.status == JobStatus.completed
    assert done.result == {"chunks": [], "metadata": {}}


async def test_complete_is_ignored_for_other_owner(queue: JobQueue):
    record = await _enqueue(queue)
    await queue.lease("worker-A")
    # A worker that does not own the lease cannot complete the job.
    await queue.complete(record.job_id, "worker-B", {"chunks": []})
    assert (await queue.get(record.job_id)).status == JobStatus.in_progress


async def test_fail_requeues_until_attempts_exhausted(queue: JobQueue):
    record = await _enqueue(queue, max_attempts=2)

    leased = await queue.lease("worker-A")
    await queue.fail(leased.job_id, "worker-A", "boom")
    requeued = await queue.get(record.job_id)
    assert requeued.status == JobStatus.scheduled
    assert requeued.error == "boom"

    leased_again = await queue.lease("worker-A")
    assert leased_again.attempts == 2
    await queue.fail(leased_again.job_id, "worker-A", "boom-again")
    assert (await queue.get(record.job_id)).status == JobStatus.failed


async def test_cancel_scheduled_job(queue: JobQueue):
    record = await _enqueue(queue)
    cancelled = await queue.cancel(record.job_id)
    assert cancelled.status == JobStatus.cancelled
    # A cancelled job is not runnable.
    assert await queue.lease("worker-A") is None


async def test_cancel_in_progress_then_complete_is_discarded(queue: JobQueue):
    record = await _enqueue(queue)
    leased = await queue.lease("worker-A")
    await queue.cancel(record.job_id)
    # Worker finishes after cancellation; the result must not resurrect the job.
    await queue.complete(leased.job_id, "worker-A", {"chunks": []})
    assert (await queue.get(record.job_id)).status == JobStatus.cancelled


async def test_reclaim_stale_returns_expired_lease_to_scheduled(queue: JobQueue):
    record = await _enqueue(queue, max_attempts=2)
    leased = await queue.lease("worker-A")
    # Force the lease to look expired.
    await queue.sql_store.update(
        queue.table_name,
        data={"lease_expires_at": int(time.time()) - 1},
        where={"job_id": leased.job_id},
    )

    reclaimed = await queue.reclaim_stale()
    assert reclaimed == 1
    assert (await queue.get(record.job_id)).status == JobStatus.scheduled


async def test_reclaim_stale_fails_job_with_no_attempts_left(queue: JobQueue):
    """An expired lease on a job that already spent its budget is terminal, not a retry."""
    record = await _enqueue(queue, max_attempts=1)
    leased = await queue.lease("worker-A")
    await queue.sql_store.update(
        queue.table_name,
        data={"lease_expires_at": int(time.time()) - 1},
        where={"job_id": leased.job_id},
    )

    assert await queue.reclaim_stale() == 1
    reclaimed = await queue.get(record.job_id)
    assert reclaimed.status == JobStatus.failed
    assert "no attempts remain" in reclaimed.error
    assert reclaimed.lease_owner is None


async def test_lease_reclaims_expired_in_progress_job(queue: JobQueue):
    record = await _enqueue(queue, max_attempts=2)
    leased = await queue.lease("worker-A")
    await queue.sql_store.update(
        queue.table_name,
        data={"lease_expires_at": int(time.time()) - 1},
        where={"job_id": leased.job_id},
    )
    # Another worker should be able to pick up the abandoned job.
    await queue.reclaim_stale()
    reclaimed = await queue.lease("worker-B")
    assert reclaimed is not None
    assert reclaimed.job_id == record.job_id
    assert reclaimed.lease_owner == "worker-B"


async def test_live_lease_expiry_fails_job_when_attempt_budget_is_exhausted(queue: JobQueue):
    record = await _enqueue(queue, max_attempts=1)
    leased = await queue.lease("worker-A")
    await queue.sql_store.update(
        queue.table_name,
        data={"lease_expires_at": int(time.time()) - 1},
        where={"job_id": leased.job_id},
    )

    await queue.reclaim_stale()
    assert await queue.lease("worker-B") is None
    failed = await queue.get(record.job_id)
    assert failed.status == JobStatus.failed
    assert failed.attempts == 1
    assert "no attempts remain" in failed.error


async def test_terminal_cleanup_is_claimed_exclusively_and_retried(queue: JobQueue):
    record = await _enqueue(queue)
    leased = await queue.lease("worker-A")
    await queue.complete(leased.job_id, "worker-A", {"chunks": []})

    claimed = await queue.claim_cleanup("cleanup-A")
    assert [item.job_id for item in claimed] == [record.job_id]
    assert await queue.claim_cleanup("cleanup-B") == []

    await queue.release_cleanup(record.job_id, "cleanup-A")
    retried = await queue.claim_cleanup("cleanup-B")
    assert [item.job_id for item in retried] == [record.job_id]

    await queue.complete_cleanup(record.job_id, "cleanup-B")
    assert await queue.claim_cleanup("cleanup-C") == []
    assert (await queue.get(record.job_id)).cleaned_at is not None


async def test_list_filters_by_api(queue: JobQueue):
    first = await _enqueue(queue, api="file_processors")
    second = await _enqueue(queue, api="file_processors")
    await _enqueue(queue, api="other")

    listed = await queue.list(api="file_processors")
    assert {r.job_id for r in listed} == {first.job_id, second.job_id}


async def test_scoped_job_operations_isolate_callers(queue: JobQueue):
    alice = User("alice", {"roles": ["member"]}, tenant_id="tenant-a")
    bob = User("bob", {"roles": ["member"]}, tenant_id="tenant-a")
    alice_job = await _enqueue(queue, authenticated_user=alice)
    bob_job = await _enqueue(queue, authenticated_user=bob)

    assert await queue.get_scoped(alice_job.job_id, "file_processors", "p1", alice) is not None
    assert await queue.get_scoped(alice_job.job_id, "file_processors", "p1", bob) is None
    assert await queue.get_scoped(alice_job.job_id, "file_processors", "other", alice) is None
    listed, has_more = await queue.list_scoped("file_processors", "p1", alice)
    assert [record.job_id for record in listed] == [alice_job.job_id]
    assert not has_more

    assert await queue.cancel_scoped(alice_job.job_id, "file_processors", "p1", bob) is None
    assert (await queue.get(alice_job.job_id)).status == JobStatus.scheduled
    assert (await queue.cancel_scoped(alice_job.job_id, "file_processors", "p1", alice)).status == JobStatus.cancelled
    assert (await queue.get(bob_job.job_id)).status == JobStatus.scheduled


async def test_scoped_job_operations_isolate_tenants_for_same_principal(queue: JobQueue):
    tenant_a = User("alice", None, tenant_id="tenant-a")
    tenant_b = User("alice", None, tenant_id="tenant-b")
    record = await _enqueue(queue, authenticated_user=tenant_a)

    assert await queue.get_scoped(record.job_id, "file_processors", "p1", tenant_b) is None
    assert await queue.get_scoped(record.job_id, "file_processors", "p1", tenant_a) is not None


async def test_scoped_job_listing_has_stable_pagination_for_equal_timestamps(queue: JobQueue):
    alice = User("alice", None, tenant_id="tenant-a")
    records = [await _enqueue(queue, authenticated_user=alice) for _ in range(5)]
    for record in records:
        await queue.sql_store.update(queue.table_name, {"created_at": 100}, where={"job_id": record.job_id})

    first_page, has_more = await queue.list_scoped("file_processors", "p1", alice, limit=2)
    second_page, second_has_more = await queue.list_scoped(
        "file_processors", "p1", alice, after=first_page[-1].job_id, limit=2
    )
    third_page, third_has_more = await queue.list_scoped(
        "file_processors", "p1", alice, after=second_page[-1].job_id, limit=2
    )

    assert has_more
    assert second_has_more
    assert not third_has_more
    assert [record.job_id for record in first_page + second_page + third_page] == sorted(
        [record.job_id for record in records], reverse=True
    )


async def test_purge_expired_removes_only_old_terminal_jobs(queue: JobQueue):
    old_completed = await _enqueue(queue)
    old_pending_cleanup = await _enqueue(queue)
    recent_failed = await _enqueue(queue)
    old_active = await _enqueue(queue)
    now = 1_000_000
    cutoff_age = 7 * 24 * 60 * 60
    await queue.sql_store.update(
        queue.table_name,
        {"status": JobStatus.completed.value, "updated_at": now - cutoff_age - 1, "cleaned_at": now - 1},
        where={"job_id": old_completed.job_id},
    )
    await queue.sql_store.update(
        queue.table_name,
        {"status": JobStatus.completed.value, "updated_at": now - cutoff_age - 1},
        where={"job_id": old_pending_cleanup.job_id},
    )
    await queue.sql_store.update(
        queue.table_name,
        {"status": JobStatus.failed.value, "updated_at": now - cutoff_age + 1, "cleaned_at": now - 1},
        where={"job_id": recent_failed.job_id},
    )
    await queue.sql_store.update(
        queue.table_name,
        {"status": JobStatus.scheduled.value, "updated_at": now - cutoff_age - 1},
        where={"job_id": old_active.job_id},
    )

    await queue.purge_expired(now=now)

    assert await queue.get(old_completed.job_id) is None
    assert await queue.get(old_pending_cleanup.job_id) is not None
    assert await queue.get(recent_failed.job_id) is not None
    assert await queue.get(old_active.job_id) is not None


async def test_second_process_must_initialize_before_querying(tmp_path):
    """A worker attaches to the server's existing table with its own empty metadata.

    SqlStore learns a table's schema only by declaring it, so a fresh process must
    call initialize() before any query — otherwise every operation raises KeyError
    on the table name. This is what the worker loop relies on at startup.
    """
    db_path = str(tmp_path / "shared-jobs.db")

    server_store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))
    server_queue = JobQueue(server_store, table_name="jobs", lease_ttl_seconds=60)
    await server_queue.initialize()
    record = await _enqueue(server_queue)

    # A distinct store over the same DB stands in for the worker process.
    worker_store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))
    worker_queue = JobQueue(worker_store, table_name="jobs", lease_ttl_seconds=60)
    with pytest.raises(KeyError):
        await worker_queue.lease("worker-A")

    await worker_queue.initialize()
    leased = await worker_queue.lease("worker-A")
    assert leased is not None
    assert leased.job_id == record.job_id

    await server_store.shutdown()
    await worker_store.shutdown()
