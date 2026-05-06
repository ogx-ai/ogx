# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import hashlib
import itertools
import json
import time
import uuid
from io import BytesIO
from typing import Any

from openai.types.batch import BatchError, Errors
from pydantic import BaseModel

from ogx.core.storage.kvstore import KVStore
from ogx.log import get_logger
from ogx_api import (
    Batches,
    BatchNotFoundError,
    BatchObject,
    ConflictError,
    Files,
    GetModelRequest,
    Inference,
    ListBatchesResponse,
    Models,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAIDeveloperMessageParam,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIFilePurpose,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from ogx_api.batches.models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)
from ogx_api.files.models import (
    RetrieveFileContentRequest,
    RetrieveFileRequest,
    UploadFileRequest,
)

from .config import ReferenceBatchesImplConfig

BATCH_PREFIX = "batch:"

logger = get_logger(__name__)


class AsyncBytesIO:
    """
    Async-compatible BytesIO wrapper to allow async file-like operations.

    We use this when uploading files to the Files API, as it expects an
    async file-like object.
    """

    def __init__(self, data: bytes):
        self._buffer = BytesIO(data)

    async def read(self, n=-1):
        return self._buffer.read(n)

    async def seek(self, pos, whence=0):
        return self._buffer.seek(pos, whence)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._buffer.close()

    def __getattr__(self, name):
        return getattr(self._buffer, name)


class BatchRequest(BaseModel):
    """Represents a single request line within a batch processing file."""

    line_num: int
    custom_id: str
    method: str
    url: str
    body: dict[str, Any]


def convert_to_openai_message_param(msg: dict[str, Any]) -> OpenAIMessageParam:
    """Convert a message dictionary to OpenAIMessageParam based on role."""
    role = msg.get("role")

    if role == "user":
        return OpenAIUserMessageParam(**msg)
    elif role == "system":
        return OpenAISystemMessageParam(**msg)
    elif role == "assistant":
        return OpenAIAssistantMessageParam(**msg)
    elif role == "tool":
        return OpenAIToolMessageParam(**msg)
    elif role == "developer":
        return OpenAIDeveloperMessageParam(**msg)
    else:
        raise ValueError(f"Unknown message role: {role}")


class ReferenceBatchesImpl(Batches):
    """Reference implementation of the Batches API.

    This implementation processes batch files by making individual requests
    to the inference API and generates output files with results.
    """

    def __init__(
        self,
        config: ReferenceBatchesImplConfig,
        inference_api: Inference,
        files_api: Files,
        models_api: Models,
        kvstore: KVStore,
    ) -> None:
        self.config = config
        self.kvstore = kvstore
        self.inference_api = inference_api
        self.files_api = files_api
        self.models_api = models_api
        self._processing_tasks: dict[str, asyncio.Task] = {}
        self._batch_semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        self._update_batch_lock = asyncio.Lock()

        # this is to allow tests to disable background processing
        self.process_batches = True

    async def initialize(self) -> None:
        # TODO: start background processing of existing tasks
        pass

    async def shutdown(self) -> None:
        """Shutdown the batches provider."""
        if self._processing_tasks:
            # don't cancel tasks - just let them stop naturally on shutdown
            # cancelling would mark batches as "cancelled" in the database
            logger.info(
                "Shutdown initiated with active batch processing tasks", active_tasks=len(self._processing_tasks)
            )

    # TODO (SECURITY): this currently works w/ configured api keys, not with x-ogx-provider-data or with user policy restrictions
    async def create_batch(
        self,
        request: CreateBatchRequest,
    ) -> BatchObject:
        """
        Create a new batch for processing multiple API requests.

        This implementation provides optional idempotency: when an idempotency key
        (idempotency_key) is provided, a deterministic ID is generated based on the input
        parameters. If a batch with the same parameters already exists, it will be
        returned instead of creating a duplicate. Without an idempotency key,
        each request creates a new batch with a unique ID.

        Args:
            input_file_id: The ID of an uploaded file containing requests for the batch.
            endpoint: The endpoint to be used for all requests in the batch.
            completion_window: The time window within which the batch should be processed.
            metadata: Optional metadata for the batch.
            idempotency_key: Optional idempotency key for enabling idempotent behavior.

        Returns:
            The created or existing batch object.
        """

        # Error handling by levels -
        #  0. Input param handling, results in 40x errors before processing, e.g.
        #    - Wrong completion_window
        #    - Invalid metadata types
        #    - Unknown endpoint
        #   -> no batch created
        #  1. Errors preventing processing, result in BatchErrors aggregated in process_batch, e.g.
        #    - input_file_id missing
        #    - invalid json in file
        #    - missing custom_id, method, url, body
        #    - invalid model
        #    - streaming
        #   -> batch created, validation sends to failed status
        #  2. Processing errors, result in error_file_id entries, e.g.
        #    - Any error returned from inference endpoint
        #   -> batch created, goes to completed status

        # TODO: set expiration time for garbage collection

        if request.endpoint not in ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]:
            raise ValueError(
                f"Invalid endpoint: {request.endpoint}. Supported values: /v1/chat/completions, /v1/completions, /v1/embeddings. Code: invalid_value. Param: endpoint",
            )

        if request.completion_window != "24h":
            raise ValueError(
                f"Invalid completion_window: {request.completion_window}. Supported values are: 24h. Code: invalid_value. Param: completion_window",
            )

        batch_id = f"batch_{uuid.uuid4().hex[:16]}"

        # For idempotent requests, use the idempotency key for the batch ID
        # This ensures the same key always maps to the same batch ID,
        # allowing us to detect parameter conflicts
        if request.idempotency_key is not None:
            hash_input = request.idempotency_key.encode("utf-8")
            hash_digest = hashlib.sha256(hash_input).hexdigest()[:24]
            batch_id = f"batch_{hash_digest}"

            try:
                existing_batch = await self.retrieve_batch(RetrieveBatchRequest(batch_id=batch_id))

                if (
                    existing_batch.input_file_id != request.input_file_id
                    or existing_batch.endpoint != request.endpoint
                    or existing_batch.completion_window != request.completion_window
                    or existing_batch.metadata != request.metadata
                ):
                    raise ConflictError(
                        f"Idempotency key '{request.idempotency_key}' was previously used with different parameters. "
                        "Either use a new idempotency key or ensure all parameters match the original request."
                    )

                logger.info("Returning existing batch", batch_id=batch_id)
                return existing_batch
            except BatchNotFoundError:
                # Batch doesn't exist, continue with creation
                pass

        current_time = int(time.time())

        batch = BatchObject(
            id=batch_id,
            object="batch",
            endpoint=request.endpoint,
            input_file_id=request.input_file_id,
            completion_window=request.completion_window,
            status="validating",
            created_at=current_time,
            metadata=request.metadata,
        )

        await self.kvstore.set(f"batch:{batch_id}", batch.to_json())
        logger.info("Created new batch", batch_id=batch_id)

        if self.process_batches:
            task = asyncio.create_task(self._process_batch(batch_id))
            self._processing_tasks[batch_id] = task

        return batch

    async def cancel_batch(self, request: CancelBatchRequest) -> BatchObject:
        """Cancel a batch that is in progress."""
        batch = await self.retrieve_batch(RetrieveBatchRequest(batch_id=request.batch_id))

        if batch.status in ["cancelled", "cancelling"]:
            return batch

        if batch.status in ["completed", "failed", "expired"]:
            raise ConflictError(f"Cannot cancel batch '{request.batch_id}' with status '{batch.status}'")

        await self._update_batch(request.batch_id, status="cancelling", cancelling_at=int(time.time()))

        if request.batch_id in self._processing_tasks:
            self._processing_tasks[request.batch_id].cancel()
            # note: task removal and status="cancelled" handled in finally block of _process_batch

        return await self.retrieve_batch(RetrieveBatchRequest(batch_id=request.batch_id))

    async def list_batches(
        self,
        request: ListBatchesRequest,
    ) -> ListBatchesResponse:
        """
        List all batches, eventually only for the current user.

        With no notion of user, we return all batches.
        """
        batch_values = await self.kvstore.values_in_range("batch:", "batch:\xff")

        batches = []
        for batch_data in batch_values:
            if batch_data:
                batches.append(BatchObject.model_validate_json(batch_data))

        batches.sort(key=lambda b: b.created_at, reverse=True)

        start_idx = 0
        if request.after:
            for i, batch in enumerate(batches):
                if batch.id == request.after:
                    start_idx = i + 1
                    break

        page_batches = batches[start_idx : start_idx + request.limit]
        has_more = (start_idx + request.limit) < len(batches)

        first_id = page_batches[0].id if page_batches else None
        last_id = page_batches[-1].id if page_batches else None

        return ListBatchesResponse(
            data=page_batches,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more,
        )

    async def retrieve_batch(self, request: RetrieveBatchRequest) -> BatchObject:
        """Retrieve information about a specific batch."""
        batch_data = await self.kvstore.get(f"batch:{request.batch_id}")
        if not batch_data:
            raise BatchNotFoundError(request.batch_id)

        return BatchObject.model_validate_json(batch_data)

    async def _update_batch(self, batch_id: str, **updates) -> None:
        """Update batch fields in kvstore."""
        async with self._update_batch_lock:
            try:
                batch = await self.retrieve_batch(RetrieveBatchRequest(batch_id=batch_id))

                # batch processing is async. once cancelling, only allow "cancelled" status updates
                if batch.status == "cancelling" and updates.get("status") != "cancelled":
                    logger.info(
                        "Skipping status update for cancelled batch",
                        batch_id=batch_id,
                        attempted_status=updates.get("status"),
                    )
                    return

                if "errors" in updates:
                    updates["errors"] = updates["errors"].model_dump()

                batch_dict = batch.model_dump()
                batch_dict.update(updates)

                await self.kvstore.set(f"batch:{batch_id}", json.dumps(batch_dict))
            except Exception as e:
                logger.error("Failed to update batch", batch_id=batch_id, error=str(e))

    async def _validate_input(self, batch: BatchObject) -> tuple[list[BatchError], list[BatchRequest]]:
        """
        Read & validate input, return errors and valid input.

        Validation of
        - input_file_id existance
        - valid json
        - custom_id, method, url, body presence and valid
        - no streaming
        """
        requests: list[BatchRequest] = []
        errors: list[BatchError] = []
        try:
            await self.files_api.openai_retrieve_file(RetrieveFileRequest(file_id=batch.input_file_id))
        except Exception:
            errors.append(
                BatchError(
                    code="invalid_request",
                    line=None,
                    message=f"Cannot find file {batch.input_file_id}.",
                    param="input_file_id",
                )
            )
            return errors, requests

        # TODO(SECURITY): do something about large files
        file_content_response = await self.files_api.openai_retrieve_file_content(
            RetrieveFileContentRequest(file_id=batch.input_file_id)
        )
        # Handle both bytes and memoryview types - convert to bytes unconditionally
        # (bytes(x) returns x if already bytes, creates new bytes from memoryview otherwise)
        body_bytes = bytes(file_content_response.body)
        file_content = body_bytes.decode("utf-8")
        for line_num, line in enumerate(file_content.strip().split("\n"), 1):
            if line.strip():  # skip empty lines
                try:
                    request = json.loads(line)

                    if not isinstance(request, dict):
                        errors.append(
                            BatchError(
                                code="invalid_request",
                                line=line_num,
                                message="Each line must be a JSON dictionary object",
                            )
                        )
                        continue

                    valid = True

                    for param, expected_type, type_string in [
                        ("custom_id", str, "string"),
                        ("method", str, "string"),
                        ("url", str, "string"),
                        ("body", dict, "JSON dictionary object"),
                    ]:
                        if param not in request:
                            errors.append(
                                BatchError(
                                    code="missing_required_parameter",
                                    line=line_num,
                                    message=f"Missing required parameter: {param}",
                                    param=param,
                                )
                            )
                            valid = False
                        elif not isinstance(request[param], expected_type):
                            param_name = "URL" if param == "url" else param.capitalize()
                            errors.append(
                                BatchError(
                                    code="invalid_request",
                                    line=line_num,
                                    message=f"{param_name} must be a {type_string}",
                                    param=param,
                                )
                            )
                            valid = False

                    if (url := request.get("url")) and isinstance(url, str) and url != batch.endpoint:
                        errors.append(
                            BatchError(
                                code="invalid_url",
                                line=line_num,
                                message="URL provided for this request does not match the batch endpoint",
                                param="url",
                            )
                        )
                        valid = False

                    if (request_body := request.get("body")) and isinstance(request_body, dict):
                        if request_body.get("stream", False):
                            errors.append(
                                BatchError(
                                    code="streaming_unsupported",
                                    line=line_num,
                                    message="Streaming is not supported in batch processing",
                                    param="body.stream",
                                )
                            )
                            valid = False

                        if batch.endpoint == "/v1/chat/completions":
                            required_params: list[tuple[str, Any, str]] = [
                                ("model", str, "a string"),
                                # messages is specific to /v1/chat/completions
                                # we could skip validating messages here and let inference fail. however,
                                # that would be a very expensive way to find out messages is wrong.
                                ("messages", list, "an array"),  # TODO: allow messages to be a string?
                            ]
                        elif batch.endpoint == "/v1/completions":
                            required_params = [
                                ("model", str, "a string"),
                                ("prompt", str, "a string"),  # TODO: allow prompt to be a list of strings??
                            ]
                        else:  # /v1/embeddings
                            required_params = [
                                ("model", str, "a string"),
                                ("input", (str, list), "a string or array of strings"),
                            ]

                        for param, expected_type, type_string in required_params:
                            if param not in request_body:
                                errors.append(
                                    BatchError(
                                        code="invalid_request",
                                        line=line_num,
                                        message=f"{param.capitalize()} parameter is required",
                                        param=f"body.{param}",
                                    )
                                )
                                valid = False
                            elif not isinstance(request_body[param], expected_type):
                                errors.append(
                                    BatchError(
                                        code="invalid_request",
                                        line=line_num,
                                        message=f"{param.capitalize()} must be {type_string}",
                                        param=f"body.{param}",
                                    )
                                )
                                valid = False

                        if "model" in request_body and isinstance(request_body["model"], str):
                            try:
                                await self.models_api.get_model(GetModelRequest(model_id=request_body["model"]))
                            except Exception:
                                errors.append(
                                    BatchError(
                                        code="model_not_found",
                                        line=line_num,
                                        message=f"Model '{request_body['model']}' does not exist or is not supported",
                                        param="body.model",
                                    )
                                )
                                valid = False

                    if valid:
                        assert isinstance(url, str), "URL must be a string"  # for mypy
                        assert isinstance(request_body, dict), "Body must be a dictionary"  # for mypy
                        requests.append(
                            BatchRequest(
                                line_num=line_num,
                                url=url,
                                method=request["method"],
                                custom_id=request["custom_id"],
                                body=request_body,
                            ),
                        )
                except json.JSONDecodeError:
                    errors.append(
                        BatchError(
                            code="invalid_json_line",
                            line=line_num,
                            message="This line is not parseable as valid JSON.",
                        )
                    )

        return errors, requests

    async def _process_batch(self, batch_id: str) -> None:
        """Background task to process a batch of requests."""
        try:
            logger.info("Starting batch processing", batch_id=batch_id)
            async with self._batch_semaphore:  # semaphore to limit concurrency
                logger.info("Acquired semaphore for batch", batch_id=batch_id)
                await self._process_batch_impl(batch_id)
        except asyncio.CancelledError:
            logger.info("Batch processing cancelled", batch_id=batch_id)
            await self._update_batch(batch_id, status="cancelled", cancelled_at=int(time.time()))
        except Exception as e:
            logger.error("Batch processing failed", batch_id=batch_id, error=str(e))
            await self._update_batch(
                batch_id,
                status="failed",
                failed_at=int(time.time()),
                errors=Errors(data=[BatchError(code="internal_error", message=str(e))]),
            )
        finally:
            self._processing_tasks.pop(batch_id, None)

    async def _process_batch_impl(self, batch_id: str) -> None:
        """Implementation of batch processing logic."""
        errors: list[BatchError] = []
        batch = await self.retrieve_batch(RetrieveBatchRequest(batch_id=batch_id))

        errors, requests = await self._validate_input(batch)
        if errors:
            await self._update_batch(batch_id, status="failed", failed_at=int(time.time()), errors=Errors(data=errors))
            logger.info("Batch validation failed", batch_id=batch_id, error_count=len(errors))
            return

        logger.info("Processing requests for batch", batch_id=batch_id, request_count=len(requests))

        total_requests = len(requests)
        await self._update_batch(
            batch_id,
            status="in_progress",
            request_counts={"total": total_requests, "completed": 0, "failed": 0},
        )

        error_results = []
        success_results = []
        completed_count = 0
        failed_count = 0

        for chunk in itertools.batched(requests, self.config.max_concurrent_requests_per_batch):
            # we use a TaskGroup to ensure all process-single-request tasks are canceled when process-batch is cancelled
            async with asyncio.TaskGroup() as tg:
                chunk_tasks = [tg.create_task(self._process_single_request(batch_id, request)) for request in chunk]

                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            for result in chunk_results:
                if isinstance(result, dict) and result.get("error") is not None:  # error response from inference
                    failed_count += 1
                    error_results.append(result)
                elif isinstance(result, dict) and result.get("response") is not None:  # successful inference
                    completed_count += 1
                    success_results.append(result)
                else:  # unexpected result
                    failed_count += 1
                    errors.append(BatchError(code="internal_error", message=f"Unexpected result: {result}"))

            await self._update_batch(
                batch_id,
                request_counts={"total": total_requests, "completed": completed_count, "failed": failed_count},
            )

            if errors:
                await self._update_batch(
                    batch_id, status="failed", failed_at=int(time.time()), errors=Errors(data=errors)
                )
                return

        try:
            output_file_id = await self._create_output_file(batch_id, success_results, "success")
            await self._update_batch(batch_id, output_file_id=output_file_id)

            error_file_id = await self._create_output_file(batch_id, error_results, "error")
            await self._update_batch(batch_id, error_file_id=error_file_id)

            await self._update_batch(batch_id, status="completed", completed_at=int(time.time()))

            logger.info(
                "Batch processing completed for : completed, failed",
                batch_id=batch_id,
                completed_count=completed_count,
                failed_count=failed_count,
            )
        except Exception as e:
            # note: errors is empty at this point, so we don't lose anything by ignoring it
            await self._update_batch(
                batch_id,
                status="failed",
                failed_at=int(time.time()),
                errors=Errors(data=[BatchError(code="output_failed", message=str(e))]),
            )

    async def _process_single_request(self, batch_id: str, request: BatchRequest) -> dict:
        """Process a single request from the batch."""
        request_id = f"batch_req_{batch_id}_{request.line_num}"

        try:
            # TODO(SECURITY): review body for security issues
            if request.url == "/v1/chat/completions":
                request.body["messages"] = [convert_to_openai_message_param(msg) for msg in request.body["messages"]]
                chat_params = OpenAIChatCompletionRequestWithExtraBody(**request.body)
                chat_response = await self.inference_api.openai_chat_completion(chat_params)

                # this is for mypy, we don't allow streaming so we'll get the right type
                assert hasattr(chat_response, "model_dump_json"), "Chat response must have model_dump_json method"
                return {
                    "id": request_id,
                    "custom_id": request.custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": request_id,  # TODO: should this be different?
                        "body": chat_response.model_dump_json(),
                    },
                }
            elif request.url == "/v1/completions":
                completion_params = OpenAICompletionRequestWithExtraBody(**request.body)
                completion_response = await self.inference_api.openai_completion(completion_params)

                # this is for mypy, we don't allow streaming so we'll get the right type
                assert hasattr(completion_response, "model_dump_json"), (
                    "Completion response must have model_dump_json method"
                )
                return {
                    "id": request_id,
                    "custom_id": request.custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": request_id,
                        "body": completion_response.model_dump_json(),
                    },
                }
            else:  # /v1/embeddings
                embeddings_response = await self.inference_api.openai_embeddings(
                    OpenAIEmbeddingsRequestWithExtraBody(**request.body)
                )
                assert hasattr(embeddings_response, "model_dump_json"), (
                    "Embeddings response must have model_dump_json method"
                )
                return {
                    "id": request_id,
                    "custom_id": request.custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": request_id,  # TODO: should this be different?
                        "body": embeddings_response.model_dump_json(),
                    },
                }
        except Exception as e:
            logger.info(
                "Error processing request in batch", custom_id=request.custom_id, batch_id=batch_id, error=str(e)
            )
            return {
                "id": request_id,
                "custom_id": request.custom_id,
                "error": {"type": "request_failed", "message": str(e)},
            }

    async def _create_output_file(self, batch_id: str, results: list[dict], file_type: str) -> str:
        """
        Create an output file with batch results.

        This function filters results based on the specified file_type
        and uploads the file to the Files API.
        """
        output_lines = [json.dumps(result) for result in results]

        with AsyncBytesIO("\n".join(output_lines).encode("utf-8")) as file_buffer:
            file_buffer.filename = f"{batch_id}_{file_type}.jsonl"
            uploaded_file = await self.files_api.openai_upload_file(
                request=UploadFileRequest(purpose=OpenAIFilePurpose.BATCH),
                file=file_buffer,
            )
            return uploaded_file.id
