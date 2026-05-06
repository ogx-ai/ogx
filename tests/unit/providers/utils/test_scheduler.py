# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import pytest

from ogx.providers.utils.scheduler import JobStatus, Scheduler


async def test_scheduler_unknown_backend():
    with pytest.raises(ValueError):
        Scheduler(backend="unknown")


async def wait_for_job_completed(sched: Scheduler, job_id: str) -> None:
    for _ in range(10):
        job = sched.get_job(job_id)
        if job.completed_at is not None:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not complete in time.")


async def test_scheduler_naive():
    sched = Scheduler()

    # make sure the scheduler starts empty
    with pytest.raises(ValueError):
        sched.get_job("unknown")
    assert sched.get_jobs() == []

    called = False

    # schedule a job that will exercise the handlers
    async def job_handler(on_log, on_status, on_artifact):
        nonlocal called
        called = True
        # exercise the handlers
        on_log("test log1")
        on_log("test log2")
        on_artifact({"type": "type1", "path": "path1"})
        on_artifact({"type": "type2", "path": "path2"})
        on_status(JobStatus.completed)

    job_id = "test_job_id"
    job_type = "test_job_type"
    sched.schedule(job_type, job_id, job_handler)

    # make sure the job was properly registered
    with pytest.raises(ValueError):
        sched.get_job("unknown")
    assert sched.get_job(job_id) is not None
    assert sched.get_jobs() == [sched.get_job(job_id)]

    assert sched.get_jobs("unknown") == []
    assert sched.get_jobs(job_type) == [sched.get_job(job_id)]

    # give the job handler a chance to run
    await wait_for_job_completed(sched, job_id)

    # now shut the scheduler down and make sure the job ran
    await sched.shutdown()

    assert called

    job = sched.get_job(job_id)
    assert job is not None

    assert job.status == JobStatus.completed

    assert job.scheduled_at is not None
    assert job.started_at is not None
    assert job.completed_at is not None
    assert job.scheduled_at < job.started_at < job.completed_at

    assert job.artifacts == [
        {"type": "type1", "path": "path1"},
        {"type": "type2", "path": "path2"},
    ]
    assert [msg[1] for msg in job.logs] == ["test log1", "test log2"]
    assert job.logs[0][0] < job.logs[1][0]


async def test_scheduler_naive_handler_raises():
    sched = Scheduler()

    async def failing_job_handler(on_log, on_status, on_artifact):
        on_status(JobStatus.running)
        raise ValueError("test error")

    job_id = "test_job_id1"
    job_type = "test_job_type"
    sched.schedule(job_type, job_id, failing_job_handler)

    job = sched.get_job(job_id)
    assert job is not None

    # confirm the exception made the job transition to failed state, even
    # though it was set to `running` before the error
    await wait_for_job_completed(sched, job_id)
    assert job.status == JobStatus.failed

    # confirm that the raised error got registered in log
    assert job.logs[0][1] == "test error"

    # even after failed job, we can schedule another one
    called = False

    async def successful_job_handler(on_log, on_status, on_artifact):
        nonlocal called
        called = True
        on_status(JobStatus.completed)

    job_id = "test_job_id2"
    sched.schedule(job_type, job_id, successful_job_handler)
    await wait_for_job_completed(sched, job_id)

    await sched.shutdown()

    assert called
    job = sched.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.completed
