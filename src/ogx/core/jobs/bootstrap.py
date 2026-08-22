# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Stack-side construction of the job runtime (queue + worker pool)."""

from ogx.core.datatypes import StackConfig
from ogx.core.storage.datatypes import SqlStoreReference
from ogx.core.storage.sqlstore.sqlstore import get_system_sqlstore

from .queue import JobQueue
from .runtime import JobRuntime, register_job_runtime
from .worker import WorkerPool

# Number of worker processes serving worker-mode (out-of-process) providers.
DEFAULT_JOB_WORKERS = 2
JOBS_TABLE = "jobs"


async def initialize_job_runtime(run_config: StackConfig) -> JobRuntime | None:
    """Build the durable job queue and worker pool for worker-mode providers.

    Returns None when there is no SQL backend to host the queue, in which case
    worker-mode providers are unavailable and will fail fast at resolution.
    """
    sql_backends = {name: cfg for name, cfg in run_config.storage.backends.items() if cfg.type.value.startswith("sql_")}
    if not sql_backends:
        return None
    backend_name = "sql_default" if "sql_default" in sql_backends else next(iter(sql_backends))

    store = await get_system_sqlstore(SqlStoreReference(backend=backend_name, table_name=JOBS_TABLE))
    queue = JobQueue(store, table_name=JOBS_TABLE)
    await queue.initialize()

    num_workers = getattr(run_config.server, "job_workers", None) or DEFAULT_JOB_WORKERS
    pool = WorkerPool(jobs_backend=backend_name, jobs_table=JOBS_TABLE, num_workers=num_workers)
    pool.set_backends(sql_backends)

    runtime = JobRuntime(queue, pool)
    register_job_runtime(runtime)
    return runtime
