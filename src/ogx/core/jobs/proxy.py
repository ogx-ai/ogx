# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Generic server-side proxy for worker-backed providers.

When a provider runs in ``worker`` mode the server mounts a proxy in place of the
real impl. The proxy does no work itself: it enqueues jobs and reads their state.
This module provides the API-agnostic machinery; each API supplies a thin
subclass that implements its own protocol methods in terms of these helpers and
maps :class:`JobRecord` to its own response types (see, e.g.,
``file_processor_proxy.py``).

Worker proxies are looked up by API via :data:`WORKER_PROXY_FACTORIES`, which each
API populates with :func:`register_worker_proxy`.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx_api import Api
from ogx_api.common.job_types import JobStatus

from .models import JobRecord
from .queue import JobQueue

logger = get_logger(name=__name__, category="core::jobs")

# How often the blocking surface polls the queue for a terminal result, and how
# long it waits before giving up.
_POLL_INTERVAL_SECONDS = 0.5
_BLOCKING_TIMEOUT_SECONDS = 1800

# The queue is unbounded and long-lived, so an uncapped list would grow without
# limit. Callers that want a different page size pass one explicitly.
DEFAULT_LIST_LIMIT = 100


class JobBackedProxy:
    """Base class for proxies that dispatch provider methods to worker processes.

    Subclasses implement their provider protocol methods using the helpers here:
    ``_enqueue`` / ``_run_blocking`` to submit work, and ``_get`` / ``_cancel`` /
    ``_list`` to expose the job lifecycle. Nothing here is API-specific.
    """

    def __init__(self, api: str, provider_id: str, job_queue: JobQueue):
        self.api = api
        self.provider_id = provider_id
        self.job_queue = job_queue

    async def _enqueue(self, method: str, payload: dict, max_attempts: int = 3) -> JobRecord:
        return await self.job_queue.enqueue(
            api=self.api,
            provider_id=self.provider_id,
            method=method,
            payload=payload,
            max_attempts=max_attempts,
            authenticated_user=get_authenticated_user(),
        )

    async def _run_blocking(
        self,
        method: str,
        payload: dict,
        timeout_seconds: int = _BLOCKING_TIMEOUT_SECONDS,
    ) -> JobRecord:
        """Enqueue a job and wait for it to finish, returning the completed record.

        Raises on failure, cancellation, or timeout. Callers map the returned
        record's ``result`` to their own response type.
        """
        record = await self._enqueue(method, payload)
        return await self._wait(record, timeout_seconds=timeout_seconds)

    async def _wait(self, record: JobRecord, timeout_seconds: int = _BLOCKING_TIMEOUT_SECONDS) -> JobRecord:
        """Poll an already-enqueued job until it reaches a terminal state.

        Split out from :meth:`_run_blocking` so callers that need to do their own
        cleanup around the enqueue (staging artifacts, say) can keep that handling
        scoped to the enqueue itself rather than to the whole wait.
        """
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            current = await self._get(record.job_id)
            if current is not None and current.is_terminal:
                if current.status == JobStatus.completed:
                    return current
                if current.status == JobStatus.cancelled:
                    raise RuntimeError(f"Failed to execute job '{record.job_id}': job was cancelled")
                raise RuntimeError(f"Failed to execute job '{record.job_id}': {current.error or 'unknown error'}")
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f"Failed to execute job '{record.job_id}': did not finish within {timeout_seconds}s")
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _get(self, job_id: str) -> JobRecord | None:
        return await self.job_queue.get_scoped(
            job_id,
            self.api,
            self.provider_id,
            get_authenticated_user(),
        )

    async def _cancel(self, job_id: str) -> JobRecord | None:
        return await self.job_queue.cancel_scoped(
            job_id,
            self.api,
            self.provider_id,
            get_authenticated_user(),
        )

    async def _list(
        self,
        after: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
    ) -> tuple[list[JobRecord], bool]:
        return await self.job_queue.list_scoped(
            self.api,
            self.provider_id,
            get_authenticated_user(),
            after=after,
            limit=limit,
        )


# A factory builds an API's proxy from the provider id, the shared queue, and the
# provider's resolved API dependencies (e.g. a file processor pulls Api.files).
WorkerProxyFactory = Callable[[str, JobQueue, dict[Api, Any]], Any]

WORKER_PROXY_FACTORIES: dict[Api, WorkerProxyFactory] = {}


def register_worker_proxy(api: Api, factory: WorkerProxyFactory) -> None:
    """Register the proxy factory used for an API's worker-mode providers."""
    WORKER_PROXY_FACTORIES[api] = factory
