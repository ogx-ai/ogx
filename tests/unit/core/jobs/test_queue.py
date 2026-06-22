# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the durable job queue."""

import time

from ogx.core.jobs.queue import JobQueue
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
    record = await _enqueue(queue)
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


async def test_lease_reclaims_expired_in_progress_job(queue: JobQueue):
    record = await _enqueue(queue)
    leased = await queue.lease("worker-A")
    await queue.sql_store.update(
        queue.table_name,
        data={"lease_expires_at": int(time.time()) - 1},
        where={"job_id": leased.job_id},
    )
    # Another worker should be able to pick up the abandoned job.
    reclaimed = await queue.lease("worker-B")
    assert reclaimed is not None
    assert reclaimed.job_id == record.job_id
    assert reclaimed.lease_owner == "worker-B"


async def test_list_filters_by_api(queue: JobQueue):
    first = await _enqueue(queue, api="file_processors")
    second = await _enqueue(queue, api="file_processors")
    await _enqueue(queue, api="other")

    listed = await queue.list(api="file_processors")
    assert {r.job_id for r in listed} == {first.job_id, second.job_id}
