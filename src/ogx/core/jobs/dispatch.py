# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Per-API (de)serialization for job payloads and results.

A worker runs a generic loop, but the arguments and return value of each
provider method are API-specific. A :class:`JobDispatcher` knows how to turn a
stored payload back into method kwargs and how to turn the method's return value
into a JSON-serializable result for storage.

Adding a new worker-backed API means adding one entry to ``JOB_DISPATCHERS``.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel


class JobDispatcher(BaseModel):
    """How to invoke and (de)serialize a single provider method run as a job."""

    model_config = {"arbitrary_types_allowed": True}

    build_kwargs: Callable[[dict[str, Any]], dict[str, Any]]
    serialize_result: Callable[[Any], dict[str, Any]]
    cleanup: Callable[[object, dict[str, Any]], Awaitable[None]] | None = None


def _file_processors_process_file_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    from ogx_api.file_processors import ProcessFileRequest

    # File bytes are never carried in the payload; the request references the
    # uploaded file by id and the worker reads it via the Files API.
    return {"request": ProcessFileRequest.model_validate(payload["request"]), "file": None}


def _serialize_pydantic(result: BaseModel) -> dict[str, Any]:
    return result.model_dump()


async def _cleanup_file_processor_upload(impl: Any, payload: dict[str, Any]) -> None:
    staged_file_id = payload.get("staged_file_id")
    if staged_file_id is None:
        return
    from ogx_api import ResourceNotFoundError
    from ogx_api.files import DeleteFileRequest

    try:
        await impl.files_api.openai_delete_file(DeleteFileRequest(file_id=staged_file_id))
    except ResourceNotFoundError:
        # A previous cleanup owner may have deleted the file before crashing
        # just short of recording completion. Absence is the desired end state.
        return


# Keyed by (api, method).
JOB_DISPATCHERS: dict[tuple[str, str], JobDispatcher] = {
    ("file_processors", "process_file"): JobDispatcher(
        build_kwargs=_file_processors_process_file_kwargs,
        serialize_result=_serialize_pydantic,
        cleanup=_cleanup_file_processor_upload,
    ),
}


def get_dispatcher(api: str, method: str) -> JobDispatcher:
    try:
        return JOB_DISPATCHERS[(api, method)]
    except KeyError as e:
        raise ValueError(f"Failed to find a job dispatcher for api='{api}' method='{method}'") from e
