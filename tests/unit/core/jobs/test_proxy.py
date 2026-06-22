# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the server-side file-processor job proxy."""

import asyncio

import pytest

from ogx.core.jobs.file_processor_proxy import FileProcessorJobProxy
from ogx.core.jobs.queue import JobQueue
from ogx_api.common.job_types import JobStatus
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.vector_io import Chunk


class _FakeUploaded:
    def __init__(self, id: str):
        self.id = id


class _FakeFilesApi:
    """Records uploads and hands back a deterministic file id."""

    def __init__(self):
        self.uploads = []

    async def openai_upload_file(self, request, file):
        self.uploads.append((request, file))
        return _FakeUploaded(id="file-staged-123")


def _result_payload() -> dict:
    return ProcessFileResponse(
        chunks=[Chunk(content="c", chunk_id="id1", chunk_metadata={})], metadata={"processor": "stub"}
    ).model_dump()


@pytest.fixture
def proxy(queue: JobQueue):
    return FileProcessorJobProxy(provider_id="p1", job_queue=queue, files_api=_FakeFilesApi())


async def test_create_job_stages_direct_upload(proxy: FileProcessorJobProxy, queue: JobQueue):
    job = await proxy.create_process_file_job(ProcessFileRequest(), file=object())
    assert job.status == JobStatus.scheduled
    assert len(proxy.files_api.uploads) == 1

    record = await queue.get(job.job_id)
    # The enqueued payload references the staged file id, never raw bytes.
    assert record.payload["request"]["file_id"] == "file-staged-123"


async def test_create_job_with_file_id_does_not_upload(proxy: FileProcessorJobProxy, queue: JobQueue):
    job = await proxy.create_process_file_job(ProcessFileRequest(file_id="existing"), file=None)
    assert proxy.files_api.uploads == []
    record = await queue.get(job.job_id)
    assert record.payload["request"]["file_id"] == "existing"


async def test_retrieve_reflects_completion(proxy: FileProcessorJobProxy, queue: JobQueue):
    job = await proxy.create_process_file_job(ProcessFileRequest(file_id="f"), file=None)
    leased = await queue.lease("worker-A")
    await queue.complete(leased.job_id, "worker-A", _result_payload())

    retrieved = await proxy.retrieve_process_file_job(job.job_id)
    assert retrieved.status == JobStatus.completed
    assert retrieved.result is not None
    assert retrieved.result.chunks[0].content == "c"


async def test_retrieve_unknown_job_raises(proxy: FileProcessorJobProxy):
    with pytest.raises(ValueError, match="Failed to find file-processing job"):
        await proxy.retrieve_process_file_job("nope")


async def test_cancel_job(proxy: FileProcessorJobProxy):
    job = await proxy.create_process_file_job(ProcessFileRequest(file_id="f"), file=None)
    cancelled = await proxy.cancel_process_file_job(job.job_id)
    assert cancelled.status == JobStatus.cancelled


async def test_list_jobs(proxy: FileProcessorJobProxy):
    await proxy.create_process_file_job(ProcessFileRequest(file_id="a"), file=None)
    await proxy.create_process_file_job(ProcessFileRequest(file_id="b"), file=None)
    listed = await proxy.list_process_file_jobs()
    assert len(listed.data) == 2


async def test_blocking_process_file_waits_for_worker(proxy: FileProcessorJobProxy, queue: JobQueue):
    """The deprecated blocking surface returns the result a worker produces."""

    async def fake_worker():
        for _ in range(50):
            leased = await queue.lease("worker-A")
            if leased is not None:
                await queue.complete(leased.job_id, "worker-A", _result_payload())
                return
            await asyncio.sleep(0.05)

    worker = asyncio.create_task(fake_worker())
    result = await proxy.process_file(ProcessFileRequest(file_id="f"), file=None)
    await worker

    assert isinstance(result, ProcessFileResponse)
    assert result.chunks[0].content == "c"


async def test_blocking_process_file_raises_on_failure(proxy: FileProcessorJobProxy, queue: JobQueue):
    async def failing_worker():
        for _ in range(50):
            leased = await queue.lease("worker-A")
            if leased is not None:
                await queue.fail(leased.job_id, "worker-A", "kaboom")
                return
            await asyncio.sleep(0.05)

    worker = asyncio.create_task(failing_worker())
    with pytest.raises(RuntimeError, match="kaboom"):
        await proxy.process_file(ProcessFileRequest(file_id="f"), file=None)
    await worker
