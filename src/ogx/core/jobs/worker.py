# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Worker process pool that executes jobs out of the server's event loop.

The server-side :class:`WorkerPool` spawns N OS processes (spawn context, so each
gets a fresh interpreter and its own GIL). Each child rebuilds the real provider
impls from :class:`ProviderDescriptor` objects, then loops: lease a job from the
shared SQL-backed queue, run the provider method, and report the result back.

Because workers talk to the same DB the server does, this same boundary makes
off-box workers reachable later without changing the control plane.
"""

import asyncio
import importlib
import multiprocessing as mp
import os
import socket
from multiprocessing.synchronize import Event as EventType

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import SqlStoreReference, StorageBackendConfig
from ogx.core.storage.sqlstore.sqlstore import get_system_sqlstore, register_sqlstore_backends
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import Api

from .dispatch import get_dispatcher
from .models import ProviderDescriptor
from .queue import JobQueue

logger = get_logger(name=__name__, category="core::jobs")

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 20


class WorkerConfig(BaseModel):
    """Everything a worker process needs to bootstrap. Pickled to the child."""

    backends: dict[str, StorageBackendConfig] = Field(default_factory=dict)
    jobs_backend: str
    jobs_table: str = "jobs"
    lease_ttl_seconds: int = 60
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    descriptors: list[ProviderDescriptor] = Field(default_factory=list)


async def _build_impl(descriptor: ProviderDescriptor) -> object:
    """Reconstruct a provider impl (and its dependencies) inside the worker."""
    deps: dict[Api, object] = {}
    for dep_api, dep_descriptor in descriptor.dependencies.items():
        deps[Api(dep_api)] = await _build_impl(dep_descriptor)

    module = importlib.import_module(descriptor.module)
    config_type = instantiate_class_type(descriptor.config_class)
    config = config_type(**descriptor.config)
    fn = getattr(module, descriptor.method)

    args: list[object] = [config, deps]
    if descriptor.pass_policy:
        # Workers execute with an empty policy; access control is enforced at the
        # server's API boundary before a job is ever enqueued.
        args.append([])
    impl = await fn(*args)
    if hasattr(impl, "initialize"):
        await impl.initialize()
    return impl


async def _heartbeat_loop(queue: JobQueue, job_id: str, worker_id: str, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=DEFAULT_HEARTBEAT_INTERVAL_SECONDS)
        except TimeoutError:
            await queue.heartbeat(job_id, worker_id)


async def _execute_job(queue: JobQueue, record, impl: object, worker_id: str) -> None:
    dispatcher = get_dispatcher(record.api, record.method)
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat_loop(queue, record.job_id, worker_id, stop))
    try:
        kwargs = dispatcher.build_kwargs(record.payload)
        method = getattr(impl, record.method)
        result = await method(**kwargs)
        await queue.complete(record.job_id, worker_id, dispatcher.serialize_result(result))
        logger.debug("Job completed", job_id=record.job_id, api=record.api, method=record.method)
    except Exception as e:
        logger.warning("Job execution failed", job_id=record.job_id, error=str(e))
        await queue.fail(record.job_id, worker_id, f"Failed to execute job: {e}")
    finally:
        stop.set()
        await heartbeat


async def _run_worker(config: WorkerConfig, stop_event: EventType) -> None:
    register_sqlstore_backends(config.backends)
    store = await get_system_sqlstore(SqlStoreReference(backend=config.jobs_backend, table_name=config.jobs_table))
    queue = JobQueue(store, config.jobs_table, config.lease_ttl_seconds)

    impls: dict[str, object] = {}
    for descriptor in config.descriptors:
        impls[descriptor.provider_id] = await _build_impl(descriptor)

    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    logger.info("Worker started", worker_id=worker_id, providers=list(impls))

    while not stop_event.is_set():
        record = await queue.lease(worker_id)
        if record is None:
            await asyncio.sleep(config.poll_interval_seconds)
            continue
        impl = impls.get(record.provider_id)
        if impl is None:
            await queue.fail(record.job_id, worker_id, f"Worker has no impl for provider '{record.provider_id}'")
            continue
        await _execute_job(queue, record, impl, worker_id)

    logger.info("Worker stopping", worker_id=worker_id)


def _worker_main(config: WorkerConfig, stop_event: EventType) -> None:
    """Process entrypoint (must be module-level for the spawn start method)."""
    try:
        asyncio.run(_run_worker(config, stop_event))
    except Exception:
        logger.error("Worker process crashed", exc_info=True)


class WorkerPool:
    """Owns and supervises the worker processes for the server.

    The server registers each worker-mode provider's descriptor, then calls
    :meth:`start` once all providers are resolved. Descriptors are accumulated
    so a single pool of processes can serve every worker-backed provider.
    """

    def __init__(self, jobs_backend: str, jobs_table: str, num_workers: int, lease_ttl_seconds: int = 60):
        self.jobs_backend = jobs_backend
        self.jobs_table = jobs_table
        self.num_workers = num_workers
        self.lease_ttl_seconds = lease_ttl_seconds
        self._backends: dict[str, StorageBackendConfig] = {}
        self._descriptors: list[ProviderDescriptor] = []
        self._processes: list[mp.process.BaseProcess] = []
        self._stop_event: EventType | None = None
        self._started = False

    def set_backends(self, backends: dict[str, StorageBackendConfig]) -> None:
        self._backends = backends

    def register(self, descriptor: ProviderDescriptor) -> None:
        self._descriptors.append(descriptor)
        logger.debug("Registered worker provider", provider_id=descriptor.provider_id, api=descriptor.api)

    def start(self) -> None:
        if self._started or not self._descriptors:
            return
        config = WorkerConfig(
            backends=self._backends,
            jobs_backend=self.jobs_backend,
            jobs_table=self.jobs_table,
            lease_ttl_seconds=self.lease_ttl_seconds,
            descriptors=self._descriptors,
        )
        ctx = mp.get_context("spawn")
        self._stop_event = ctx.Event()
        for _ in range(self.num_workers):
            proc = ctx.Process(target=_worker_main, args=(config, self._stop_event), daemon=True)
            proc.start()
            self._processes.append(proc)
        self._started = True
        logger.info("Started worker pool", num_workers=self.num_workers, providers=len(self._descriptors))

    def shutdown(self, timeout: float = 10.0) -> None:
        if not self._started:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        for proc in self._processes:
            proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
        self._processes.clear()
        self._started = False
        logger.info("Worker pool shut down")
