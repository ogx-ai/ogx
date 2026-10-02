# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Process-wide handle to the job queue and worker pool.

The queue and pool are owned by the stack but needed during provider resolution
(to hand the proxy a queue and to register worker descriptors). Rather than
thread them through every resolver signature, they are registered here as a
process-global — mirroring the existing storage-backend registry pattern in
:mod:`ogx.core.storage.sqlstore.sqlstore`.
"""

from .queue import JobQueue
from .worker import WorkerPool


class JobRuntime:
    """Bundles the shared queue and worker pool for a running stack."""

    def __init__(self, queue: JobQueue, pool: WorkerPool):
        self.queue = queue
        self.pool = pool


_RUNTIME: JobRuntime | None = None


def register_job_runtime(runtime: JobRuntime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def get_job_runtime() -> JobRuntime | None:
    return _RUNTIME


def reset_job_runtime() -> None:
    global _RUNTIME
    _RUNTIME = None
