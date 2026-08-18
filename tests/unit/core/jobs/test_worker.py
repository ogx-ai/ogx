# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for worker job execution (the in-process pieces of the worker loop)."""

import asyncio

import pytest

from ogx.core.datatypes import User
from ogx.core.jobs import worker
from ogx.core.jobs.queue import JobQueue
from ogx.core.jobs.worker import WorkerConfig, WorkerPool, _execute_job, _worker_main
from ogx.core.request_headers import get_authenticated_user
from ogx_api import ResourceNotFoundError
from ogx_api.common.job_types import JobStatus
from ogx_api.file_processors import ProcessFileResponse
from ogx_api.vector_io import Chunk


class _StubProcessor:
    """Minimal file processor that records its calls and returns a fixed result."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []
        self.seen_user = None
        self.files_api = _FakeFilesApi()

    async def process_file(self, request, file=None):
        self.calls.append(request)
        self.seen_user = get_authenticated_user()
        if self.fail:
            raise RuntimeError("processing exploded")
        return ProcessFileResponse(
            chunks=[Chunk(content="hello", chunk_id="id1", chunk_metadata={})],
            metadata={"processor": "stub"},
        )


class _FakeFilesApi:
    def __init__(self):
        self.deleted = []
        self.cleanup_users = []

    async def openai_delete_file(self, request):
        self.deleted.append(request.file_id)
        self.cleanup_users.append(get_authenticated_user())


async def _lease_one(
    queue: JobQueue,
    authenticated_user: User | None = None,
    payload: dict | None = None,
    max_attempts: int = 1,
):
    record = await queue.enqueue(
        api="file_processors",
        provider_id="p1",
        method="process_file",
        payload=payload or {"request": {"file_id": "f1"}},
        authenticated_user=authenticated_user,
        max_attempts=max_attempts,
    )
    leased = await queue.lease("worker-A")
    assert leased is not None and leased.job_id == record.job_id
    return leased


async def test_execute_job_completes_and_serializes_result(queue: JobQueue):
    leased = await _lease_one(queue)
    impl = _StubProcessor()

    await _execute_job(queue, leased, impl, "worker-A")

    assert len(impl.calls) == 1
    assert impl.calls[0].file_id == "f1"

    done = await queue.get(leased.job_id)
    assert done.status == JobStatus.completed
    result = ProcessFileResponse.model_validate(done.result)
    assert result.chunks[0].content == "hello"
    assert result.metadata == {"processor": "stub"}


async def test_execute_job_marks_failed_on_exception(queue: JobQueue):
    leased = await _lease_one(queue)
    impl = _StubProcessor(fail=True)

    await _execute_job(queue, leased, impl, "worker-A")

    done = await queue.get(leased.job_id)
    assert done.status == JobStatus.failed
    assert "processing exploded" in done.error


async def test_execute_job_restores_authenticated_user(queue: JobQueue):
    user = User("alice", {"roles": ["member"]}, tenant_id="tenant-a")
    leased = await _lease_one(queue, authenticated_user=user)
    impl = _StubProcessor()

    await _execute_job(queue, leased, impl, "worker-A")

    assert impl.seen_user == user


async def test_execute_job_cleans_staged_file_after_success(queue: JobQueue):
    user = User("alice", {"roles": ["member"]}, tenant_id="tenant-a")
    leased = await _lease_one(
        queue,
        authenticated_user=user,
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
    )
    impl = _StubProcessor()

    await _execute_job(queue, leased, impl, "worker-A")

    assert impl.files_api.deleted == ["file-staged"]
    assert impl.files_api.cleanup_users == [user]


async def test_reclaimed_final_attempt_cleans_staged_file(queue: JobQueue):
    user = User("alice", {"roles": ["member"]}, tenant_id="tenant-a")
    leased = await _lease_one(
        queue,
        authenticated_user=user,
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
    )
    await queue.sql_store.update(
        queue.table_name,
        data={"lease_expires_at": 0},
        where={"job_id": leased.job_id},
    )
    impl = _StubProcessor()

    await worker._run_maintenance(
        queue,
        {("file_processors", "p1"): impl},
        purge_expired=False,
        worker_id="worker-B",
    )

    assert (await queue.get(leased.job_id)).status == JobStatus.failed
    assert impl.files_api.deleted == ["file-staged"]
    assert impl.files_api.cleanup_users == [user]


async def test_worker_maintenance_purges_expired_terminal_jobs(queue: JobQueue):
    record = await queue.enqueue(
        api="file_processors",
        provider_id="p1",
        method="process_file",
        payload={"request": {"file_id": "existing"}},
    )
    await queue.sql_store.update(
        queue.table_name,
        data={"status": JobStatus.completed.value, "updated_at": 0},
        where={"job_id": record.job_id},
    )

    await worker._run_maintenance(
        queue,
        {("file_processors", "p1"): _StubProcessor()},
        purge_expired=True,
    )

    assert await queue.get(record.job_id) is None


async def test_execute_job_keeps_staged_file_while_retry_is_scheduled(queue: JobQueue):
    leased = await _lease_one(
        queue,
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
        max_attempts=2,
    )
    impl = _StubProcessor(fail=True)

    await _execute_job(queue, leased, impl, "worker-A")

    assert (await queue.get(leased.job_id)).status == JobStatus.scheduled
    assert impl.files_api.deleted == []


async def test_execute_job_cleans_staged_file_after_permanent_failure(queue: JobQueue):
    leased = await _lease_one(
        queue,
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
    )
    impl = _StubProcessor(fail=True)

    await _execute_job(queue, leased, impl, "worker-A")

    assert (await queue.get(leased.job_id)).status == JobStatus.failed
    assert impl.files_api.deleted == ["file-staged"]


async def test_failed_terminal_cleanup_is_retried_by_maintenance(queue: JobQueue):
    leased = await _lease_one(
        queue,
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
    )
    impl = _StubProcessor()
    original_delete = impl.files_api.openai_delete_file
    attempts = 0

    async def flaky_delete(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary cleanup failure")
        await original_delete(request)

    impl.files_api.openai_delete_file = flaky_delete

    await _execute_job(queue, leased, impl, "worker-A")
    assert (await queue.get(leased.job_id)).cleaned_at is None

    await worker._run_maintenance(
        queue,
        {("file_processors", "p1"): impl},
        purge_expired=False,
        worker_id="worker-B",
    )

    assert impl.files_api.deleted == ["file-staged"]
    assert (await queue.get(leased.job_id)).cleaned_at is not None


async def test_cancelled_staged_upload_is_cleaned_by_maintenance(queue: JobQueue):
    record = await queue.enqueue(
        api="file_processors",
        provider_id="p1",
        method="process_file",
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
    )
    await queue.cancel(record.job_id)
    impl = _StubProcessor()

    await worker._run_maintenance(
        queue,
        {("file_processors", "p1"): impl},
        purge_expired=False,
        worker_id="worker-A",
    )

    assert impl.files_api.deleted == ["file-staged"]
    assert (await queue.get(record.job_id)).cleaned_at is not None


async def test_cleanup_marks_already_deleted_staged_upload_complete(queue: JobQueue):
    record = await queue.enqueue(
        api="file_processors",
        provider_id="p1",
        method="process_file",
        payload={"request": {"file_id": "file-staged"}, "staged_file_id": "file-staged"},
    )
    await queue.sql_store.update(
        queue.table_name,
        data={"status": JobStatus.completed.value},
        where={"job_id": record.job_id},
    )
    impl = _StubProcessor()

    async def already_deleted(_request):
        raise ResourceNotFoundError("file-staged", resource_type="File")

    impl.files_api.openai_delete_file = already_deleted

    await worker._run_maintenance(
        queue,
        {("file_processors", "p1"): impl},
        purge_expired=False,
        worker_id="worker-A",
    )

    assert (await queue.get(record.job_id)).cleaned_at is not None


def test_worker_main_reraises_fatal_error(monkeypatch) -> None:
    def fail(coroutine):
        coroutine.close()
        raise RuntimeError("bootstrap exploded")

    monkeypatch.setattr(asyncio, "run", fail)

    with pytest.raises(RuntimeError, match="bootstrap exploded"):
        _worker_main(WorkerConfig(jobs_backend="sql_default"), object())


class _FakeProcess:
    def __init__(self, alive: bool):
        self.alive = alive
        self.joined = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.joined = True

    def terminate(self):
        self.alive = False


class _FakeStopEvent:
    def __init__(self, stopped: bool = False):
        self.stopped = stopped

    def is_set(self):
        return self.stopped


def test_worker_pool_restarts_dead_slots_with_backoff(monkeypatch) -> None:
    pool = WorkerPool(jobs_backend="sql_default", jobs_table="jobs", num_workers=2)
    dead = _FakeProcess(alive=False)
    healthy = _FakeProcess(alive=True)
    replacement = _FakeProcess(alive=True)
    pool._processes = [dead, healthy]
    pool._restart_failures = [0, 0]
    pool._next_restart_at = [0.0, 0.0]
    pool._started_at = [0.0, 0.0]
    pool._stop_event = _FakeStopEvent()
    pool._started = True
    spawned = []

    def spawn():
        spawned.append(replacement)
        return replacement

    monkeypatch.setattr(pool, "_spawn_process", spawn)

    pool._maintain_workers(now=10.0)

    assert dead.joined
    assert pool._processes == [replacement, healthy]
    assert spawned == [replacement]
    assert pool.is_healthy

    replacement.alive = False
    pool._maintain_workers(now=10.5)
    assert spawned == [replacement]


def test_worker_pool_does_not_restart_after_stop(monkeypatch) -> None:
    pool = WorkerPool(jobs_backend="sql_default", jobs_table="jobs", num_workers=1)
    pool._processes = [_FakeProcess(alive=False)]
    pool._restart_failures = [0]
    pool._next_restart_at = [0.0]
    pool._started_at = [0.0]
    pool._stop_event = _FakeStopEvent(stopped=True)
    pool._started = True

    monkeypatch.setattr(pool, "_spawn_process", lambda: pytest.fail("worker restarted during shutdown"))

    pool._maintain_workers(now=10.0)
    assert not pool.is_healthy


def test_worker_pool_without_registered_workers_is_healthy() -> None:
    pool = WorkerPool(jobs_backend="sql_default", jobs_table="jobs", num_workers=1)

    assert pool.is_healthy


def test_worker_pool_cleans_up_partial_start_failure(monkeypatch) -> None:
    pool = WorkerPool(jobs_backend="sql_default", jobs_table="jobs", num_workers=2)
    pool.register(
        worker.ProviderDescriptor(
            api="file_processors",
            provider_id="p1",
            provider_type="inline::pypdf",
            module="provider.module",
            config_class="provider.Config",
        )
    )
    first = _FakeProcess(alive=True)
    calls = 0

    def spawn():
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(pool, "_spawn_process", spawn)

    with pytest.raises(RuntimeError, match="spawn failed"):
        pool.start()

    assert first.joined
    assert not first.alive
    assert pool._processes == []
    assert not pool._started
