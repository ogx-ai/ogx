# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from fastapi import UploadFile

from .models import ListProcessFileJobsResponse, ProcessFileJob, ProcessFileRequest, ProcessFileResponse


@runtime_checkable
class FileProcessors(Protocol):
    """
    File Processor API for converting files into structured, processable content.

    This API provides a flexible interface for processing various file formats
    (PDFs, documents, images, etc.) into normalized text content that can be used for
    vector store ingestion, RAG applications, or standalone content extraction.

    The API focuses on parsing and normalization:
    - Multiple file formats through extensible provider architecture
    - Multipart form uploads or file ID references
    - Configurable processing options per provider
    - Optional chunking using provider's native capabilities
    - Rich metadata about processing results

    For embedding generation, use the chunks from this API with the separate
    embedding API to maintain clean separation of concerns.

    Future providers can extend this interface to support additional formats,
    processing capabilities, and optimization strategies.
    """

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """
        Process a file into chunks ready for vector database storage.

        This method supports two modes of operation via multipart form request:
        1. Direct upload: Upload and process a file directly (file parameter)
        2. File storage: Process files already uploaded to file storage (request.file_id parameter)

        Exactly one of file or request.file_id must be provided.

        If no chunking_strategy is provided, the entire file content is returned as a single chunk.
        If chunking_strategy is provided, the file is split according to the strategy.

        :param request: The request containing file_id, options, and chunking_strategy.
        :param file: The uploaded file object containing content and metadata (filename, content_type, etc.). Mutually exclusive with request.file_id.
        :returns: ProcessFileResponse with chunks ready for vector database storage.
        """
        ...

    async def create_process_file_job(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileJob:
        """
        Submit a file for asynchronous processing and return immediately.

        Unlike process_file, this does not block until processing finishes. The
        work runs out-of-process; poll retrieve_process_file_job with the returned
        job_id until the status is terminal.

        :param request: The request containing file_id, options, and chunking_strategy.
        :param file: The uploaded file object. Mutually exclusive with request.file_id.
        :returns: A ProcessFileJob in the 'scheduled' state.
        """
        ...

    async def retrieve_process_file_job(self, job_id: str) -> ProcessFileJob:
        """
        Retrieve the current state of a file-processing job.

        :param job_id: The id of the job to retrieve.
        :returns: The ProcessFileJob, including the result once completed.
        """
        ...

    async def cancel_process_file_job(self, job_id: str) -> ProcessFileJob:
        """
        Cancel a scheduled or in-progress file-processing job.

        :param job_id: The id of the job to cancel.
        :returns: The ProcessFileJob in its post-cancellation state.
        """
        ...

    async def list_process_file_jobs(self) -> ListProcessFileJobsResponse:
        """
        List file-processing jobs, most recent first.

        :returns: The list of ProcessFileJob records.
        """
        ...
