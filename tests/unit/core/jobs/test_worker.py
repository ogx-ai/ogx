# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for worker job execution (the in-process pieces of the worker loop)."""

from ogx.core.jobs.queue import JobQueue
from ogx.core.jobs.worker import _execute_job
from ogx_api.common.job_types import JobStatus
from ogx_api.file_processors import ProcessFileResponse
from ogx_api.vector_io import Chunk


class _StubProcessor:
    """Minimal file processor that records its calls and returns a fixed result."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    async def process_file(self, request, file=None):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("processing exploded")
        return ProcessFileResponse(
            chunks=[Chunk(content="hello", chunk_id="id1", chunk_metadata={})],
            metadata={"processor": "stub"},
        )


async def _lease_one(queue: JobQueue):
    record = await queue.enqueue(
        api="file_processors",
        provider_id="p1",
        method="process_file",
        payload={"request": {"file_id": "f1"}},
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
