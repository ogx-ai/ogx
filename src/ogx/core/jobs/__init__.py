# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Generic out-of-process job execution substrate.

This package provides a durable job queue and a worker process pool that let
"inline" providers run their work in a separate OS process instead of the
server's event loop. Providers opt in via ``execution_mode: worker`` on their
``InlineProviderSpec``; the resolver then hands the server a thin proxy impl
(see :mod:`ogx.core.jobs.proxy`) that enqueues work onto the queue, while the
real provider impl runs inside a worker (see :mod:`ogx.core.jobs.worker`).

The queue is backed by the same SQL store used elsewhere in OGX, so jobs are
durable across restarts and the queue table doubles as the IPC channel between
the server and its workers.
"""

from . import file_processor_proxy  # noqa: F401  imported for its register_worker_proxy side effect
from .file_processor_proxy import FileProcessorJobProxy
from .models import JobRecord, ProviderDescriptor
from .proxy import WORKER_PROXY_FACTORIES, JobBackedProxy, register_worker_proxy
from .queue import JobQueue
from .worker import WorkerPool

__all__ = [
    "WORKER_PROXY_FACTORIES",
    "FileProcessorJobProxy",
    "JobBackedProxy",
    "JobQueue",
    "JobRecord",
    "ProviderDescriptor",
    "WorkerPool",
    "register_worker_proxy",
]
