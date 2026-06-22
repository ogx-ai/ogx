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

from ogx_api import Api, Files
from ogx_api.file_processors import (
    ListProcessFileJobsResponse,
    ProcessFileJob,
    ProcessFileRequest,
    ProcessFileResponse,
)
from ogx_api.files import OpenAIFileUploadPurpose, UploadFileRequest

from .models import JobRecord
from .proxy import JobBackedProxy, register_worker_proxy
from .queue import JobQueue

PROCESS_FILE_METHOD = "process_file"


class FileProcessorJobProxy(JobBackedProxy):
    """A ``FileProcessors`` impl that runs work in worker processes."""

    def __init__(self, provider_id: str, job_queue: JobQueue, files_api: Files):
        super().__init__(api=Api.file_processors.value, provider_id=provider_id, job_queue=job_queue)
        self.files_api = files_api

    async def _build_payload(self, request: ProcessFileRequest, file: UploadFile | None) -> dict:
        """Stage a direct upload into Files (so the worker can read it by id) and serialize."""
        if file is not None:
            uploaded = await self.files_api.openai_upload_file(
                UploadFileRequest(purpose=OpenAIFileUploadPurpose.USER_DATA),
                file,
            )
            request = request.model_copy(update={"file_id": uploaded.id})
        return {"request": request.model_dump(mode="json")}

    @staticmethod
    def _to_job(record: JobRecord) -> ProcessFileJob:
        result = ProcessFileResponse.model_validate(record.result) if record.result is not None else None
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
        record = await self._run_blocking(PROCESS_FILE_METHOD, await self._build_payload(request, file))
        return ProcessFileResponse.model_validate(record.result)

    async def create_process_file_job(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileJob:
        record = await self._enqueue(PROCESS_FILE_METHOD, await self._build_payload(request, file))
        return self._to_job(record)

    async def retrieve_process_file_job(self, job_id: str) -> ProcessFileJob:
        record = await self._get(job_id)
        if record is None:
            raise ValueError(f"Failed to find file-processing job '{job_id}'")
        return self._to_job(record)

    async def cancel_process_file_job(self, job_id: str) -> ProcessFileJob:
        record = await self._cancel(job_id)
        if record is None:
            raise ValueError(f"Failed to find file-processing job '{job_id}'")
        return self._to_job(record)

    async def list_process_file_jobs(self) -> ListProcessFileJobsResponse:
        return ListProcessFileJobsResponse(data=[self._to_job(r) for r in await self._list()])


def _file_processor_proxy_factory(provider_id: str, job_queue: JobQueue, deps: dict[Api, Any]) -> FileProcessorJobProxy:
    return FileProcessorJobProxy(provider_id=provider_id, job_queue=job_queue, files_api=deps[Api.files])


register_worker_proxy(Api.file_processors, _file_processor_proxy_factory)
