# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the server-side file-processor job proxy."""

import asyncio

import pytest

from ogx.core.datatypes import User
from ogx.core.jobs.file_processor_proxy import FileProcessorJobProxy
from ogx.core.jobs.queue import JobQueue
from ogx.core.request_headers import RequestProviderDataContext
from ogx_api import ResourceNotFoundError
from ogx_api.common.job_types import JobStatus
from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse
from ogx_api.vector_io import Chunk


class _FakeUploaded:
    def __init__(self, id: str):
        self.id = id


class _FakeFilesApi:
    """Records uploads and deletions, and hands back a deterministic file id."""

    def __init__(self):
        self.uploads = []
        self.deleted = []

    async def openai_upload_file(self, request, file):
        self.uploads.append((request, file))
        return _FakeUploaded(id="file-staged-123")

    async def openai_delete_file(self, request):
        self.deleted.append(request.file_id)


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
    assert record.payload["staged_file_id"] == "file-staged-123"


@pytest.mark.parametrize(
    ("process_request", "file"),
    [
        (ProcessFileRequest(), None),
        (ProcessFileRequest(file_id="existing"), object()),
    ],
)
async def test_direct_proxy_calls_require_exactly_one_input(
    proxy: FileProcessorJobProxy,
    process_request: ProcessFileRequest,
    file: object | None,
) -> None:
    with pytest.raises(ValueError, match="exactly one of file or file_id"):
        await proxy.create_process_file_job(process_request, file)

    assert proxy.files_api.uploads == []


async def test_failed_enqueue_cleans_up_staged_upload(proxy: FileProcessorJobProxy, queue: JobQueue):
    """A staged upload must not be orphaned when the job never reaches the queue."""

    async def boom(*args, **kwargs):
        raise RuntimeError("db is down")

    queue.enqueue = boom

    with pytest.raises(RuntimeError, match="db is down"):
        await proxy.create_process_file_job(ProcessFileRequest(), file=object())

    assert proxy.files_api.deleted == ["file-staged-123"]


async def test_failed_enqueue_without_upload_deletes_nothing(proxy: FileProcessorJobProxy, queue: JobQueue):
    async def boom(*args, **kwargs):
        raise RuntimeError("db is down")

    queue.enqueue = boom

    with pytest.raises(RuntimeError, match="db is down"):
        await proxy.create_process_file_job(ProcessFileRequest(file_id="existing"), file=None)

    assert proxy.files_api.deleted == []


async def test_create_job_with_file_id_does_not_upload(proxy: FileProcessorJobProxy, queue: JobQueue):
    job = await proxy.create_process_file_job(ProcessFileRequest(file_id="existing"), file=None)
    assert proxy.files_api.uploads == []
    record = await queue.get(job.job_id)
    assert record.payload["request"]["file_id"] == "existing"
    assert record.max_attempts == 3


async def test_retrieve_reflects_completion(proxy: FileProcessorJobProxy, queue: JobQueue):
    job = await proxy.create_process_file_job(ProcessFileRequest(file_id="f"), file=None)
    leased = await queue.lease("worker-A")
    await queue.complete(leased.job_id, "worker-A", _result_payload())

    retrieved = await proxy.retrieve_process_file_job(job.job_id)
    assert retrieved.status == JobStatus.completed
    assert retrieved.result is not None
    assert retrieved.result.chunks[0].content == "c"


async def test_retrieve_unknown_job_raises(proxy: FileProcessorJobProxy):
    with pytest.raises(ResourceNotFoundError, match="File-processing job 'nope' not found"):
        await proxy.retrieve_process_file_job("nope")


async def test_cancel_unknown_job_raises_not_found(proxy: FileProcessorJobProxy):
    with pytest.raises(ResourceNotFoundError, match="File-processing job 'nope' not found"):
        await proxy.cancel_process_file_job("nope")


async def test_cancel_job_leaves_staged_file_for_durable_cleanup(proxy: FileProcessorJobProxy, queue: JobQueue):
    job = await proxy.create_process_file_job(ProcessFileRequest(), file=object())
    cancelled = await proxy.cancel_process_file_job(job.job_id)
    assert cancelled.status == JobStatus.cancelled
    assert proxy.files_api.deleted == []
    assert (await queue.get(job.job_id)).cleaned_at is None


async def test_list_jobs(proxy: FileProcessorJobProxy):
    await proxy.create_process_file_job(ProcessFileRequest(file_id="a"), file=None)
    await proxy.create_process_file_job(ProcessFileRequest(file_id="b"), file=None)
    listed = await proxy.list_process_file_jobs()
    assert len(listed.data) == 2
    assert not listed.has_more


async def test_list_jobs_is_paginated_and_omits_result_bodies(proxy: FileProcessorJobProxy, queue: JobQueue):
    jobs = [
        await proxy.create_process_file_job(ProcessFileRequest(file_id=file_id), file=None)
        for file_id in ("a", "b", "c")
    ]
    leased = await queue.lease("worker-A")
    await queue.complete(leased.job_id, "worker-A", _result_payload())

    first_page = await proxy.list_process_file_jobs(limit=2)
    second_page = await proxy.list_process_file_jobs(after=first_page.data[-1].job_id, limit=2)

    assert first_page.has_more
    assert not second_page.has_more
    assert len(first_page.data) == 2
    assert len(second_page.data) == 1
    assert all(job.result is None for job in first_page.data + second_page.data)
    assert {job.job_id for job in first_page.data + second_page.data} == {job.job_id for job in jobs}


async def test_proxy_job_control_is_scoped_to_authenticated_caller(proxy: FileProcessorJobProxy):
    alice = User("alice", {"roles": ["member"]}, tenant_id="tenant-a")
    bob = User("bob", {"roles": ["member"]}, tenant_id="tenant-a")
    with RequestProviderDataContext(user=alice):
        alice_job = await proxy.create_process_file_job(ProcessFileRequest(file_id="a"), file=None)

    with RequestProviderDataContext(user=bob):
        with pytest.raises(ResourceNotFoundError, match="File-processing job"):
            await proxy.retrieve_process_file_job(alice_job.job_id)
        assert (await proxy.list_process_file_jobs()).data == []
        with pytest.raises(ResourceNotFoundError, match="File-processing job"):
            await proxy.cancel_process_file_job(alice_job.job_id)

    with RequestProviderDataContext(user=alice):
        assert (await proxy.retrieve_process_file_job(alice_job.job_id)).job_id == alice_job.job_id


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
        failed_attempts = 0
        while failed_attempts < 3:
            leased = await queue.lease("worker-A")
            if leased is None:
                await asyncio.sleep(0.05)
                continue
            await queue.fail(leased.job_id, "worker-A", "kaboom")
            failed_attempts += 1

    worker = asyncio.create_task(failing_worker())
    with pytest.raises(RuntimeError, match="kaboom"):
        await proxy.process_file(ProcessFileRequest(file_id="f"), file=None)
    await worker
