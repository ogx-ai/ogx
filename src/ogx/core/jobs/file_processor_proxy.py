# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""File-processor adapter for the generic worker proxy.

This is the only file-processor-specific part of the worker path. It maps the
``FileProcessors`` protocol onto the generic :class:`JobBackedProxy`: it knows the
unit of work is ``process_file``, how to turn a request (and an optional direct
upload) into a serializable payload, and how to render a :class:`JobRecord` as a
``ProcessFileJob``. Everything else is inherited.

Other APIs add worker support by writing an analogous adapter and calling
:func:`register_worker_proxy` — no changes to the engine, resolver, or worker.
"""

from typing import Any

from fastapi import UploadFile

from ogx.log import get_logger
from ogx_api import Api, Files, ResourceNotFoundError
from ogx_api.file_processors import (
    ListProcessFileJobsResponse,
    ProcessFileJob,
    ProcessFileRequest,
    ProcessFileResponse,
)
from ogx_api.files import DeleteFileRequest, OpenAIFileUploadPurpose, UploadFileRequest

from .models import JobRecord
from .proxy import JobBackedProxy, register_worker_proxy
from .queue import JobQueue

logger = get_logger(name=__name__, category="core::jobs")

PROCESS_FILE_METHOD = "process_file"


class FileProcessorJobProxy(JobBackedProxy):
    """A ``FileProcessors`` impl that runs work in worker processes."""

    def __init__(self, provider_id: str, job_queue: JobQueue, files_api: Files):
        super().__init__(api=Api.file_processors.value, provider_id=provider_id, job_queue=job_queue)
        self.files_api = files_api

    async def _submit(self, request: ProcessFileRequest, file: UploadFile | None) -> JobRecord:
        """Stage a direct upload into Files (so the worker can read it by id), then enqueue.

        Staging happens before the enqueue, so a failed enqueue would otherwise
        leave the uploaded file behind with nothing referencing it. Cleanup is
        scoped to that window only: once the job is queued the file belongs to the
        job and must outlive this call.
        """
        if (file is None) == (request.file_id is None):
            raise ValueError("Failed to process file: exactly one of file or file_id must be provided.")

        staged_file_id: str | None = None
        if file is not None:
            uploaded = await self.files_api.openai_upload_file(
                UploadFileRequest(purpose=OpenAIFileUploadPurpose.USER_DATA),
                file,
            )
            staged_file_id = uploaded.id
            request = request.model_copy(update={"file_id": uploaded.id})

        try:
            payload: dict[str, Any] = {"request": request.model_dump(mode="json")}
            if staged_file_id is not None:
                payload["staged_file_id"] = staged_file_id
            return await self._enqueue(PROCESS_FILE_METHOD, payload)
        except Exception:
            if staged_file_id is not None:
                await self._discard_staged_file(staged_file_id)
            raise

    async def _discard_staged_file(self, file_id: str) -> None:
        """Best-effort removal of a staged upload whose job never made it onto the queue."""
        try:
            await self.files_api.openai_delete_file(DeleteFileRequest(file_id=file_id))
        except Exception as e:
            logger.warning("Failed to delete orphaned staged file", file_id=file_id, error=str(e))

    @staticmethod
    def _to_job(record: JobRecord, include_result: bool = True) -> ProcessFileJob:
        result = (
            ProcessFileResponse.model_validate(record.result) if include_result and record.result is not None else None
        )
        return ProcessFileJob(
            job_id=record.job_id,
            status=record.status,
            created_at=record.created_at,
            result=result,
            error=record.error,
        )

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Deprecated blocking surface: enqueue and wait for the worker's result."""
        completed = await self._wait(await self._submit(request, file))
        return ProcessFileResponse.model_validate(completed.result)

    async def create_process_file_job(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileJob:
        return self._to_job(await self._submit(request, file))

    async def retrieve_process_file_job(self, job_id: str) -> ProcessFileJob:
        record = await self._get(job_id)
        if record is None:
            raise ResourceNotFoundError(job_id, resource_type="File-processing job")
        return self._to_job(record)

    async def cancel_process_file_job(self, job_id: str) -> ProcessFileJob:
        record = await self._cancel(job_id)
        if record is None:
            raise ResourceNotFoundError(job_id, resource_type="File-processing job")
        return self._to_job(record)

    async def list_process_file_jobs(
        self,
        after: str | None = None,
        limit: int = 100,
    ) -> ListProcessFileJobsResponse:
        records, has_more = await self._list(after=after, limit=limit)
        return ListProcessFileJobsResponse(
            data=[self._to_job(record, include_result=False) for record in records],
            has_more=has_more,
        )


def _file_processor_proxy_factory(provider_id: str, job_queue: JobQueue, deps: dict[Api, Any]) -> FileProcessorJobProxy:
    return FileProcessorJobProxy(provider_id=provider_id, job_queue=job_queue, files_api=deps[Api.files])


register_worker_proxy(Api.file_processors, _file_processor_proxy_factory)
