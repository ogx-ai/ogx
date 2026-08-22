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
import threading
import time
import traceback
from multiprocessing.context import SpawnContext
from multiprocessing.synchronize import Event as EventType

from pydantic import BaseModel, Field

from ogx.core.datatypes import User
from ogx.core.request_headers import RequestProviderDataContext
from ogx.core.storage.datatypes import SqlStoreReference, StorageBackendConfig
from ogx.core.storage.sqlstore.sqlstore import get_system_sqlstore, register_sqlstore_backends
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import Api

from .dispatch import get_dispatcher
from .models import JobRecord, ProviderDescriptor
from .queue import JobQueue

logger = get_logger(name=__name__, category="core::jobs")

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 20
DEFAULT_MAINTENANCE_INTERVAL_SECONDS = 300


class WorkerConfig(BaseModel):
    """Everything a worker process needs to bootstrap. Pickled to the child."""

    backends: dict[str, StorageBackendConfig] = Field(default_factory=dict)
    jobs_backend: str
    jobs_table: str = "jobs"
    lease_ttl_seconds: int = 60
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    maintenance_interval_seconds: float = DEFAULT_MAINTENANCE_INTERVAL_SECONDS
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
        args.append(descriptor.policy)
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


def _wire_sibling_providers(impls: dict[tuple[str, str], object]) -> None:
    """Inject complete same-API sibling sets after all root providers are built."""
    by_api: dict[str, dict[str, object]] = {}
    for (api, provider_id), impl in impls.items():
        by_api.setdefault(api, {})[provider_id] = impl

    for siblings in by_api.values():
        for provider_id, impl in siblings.items():
            if hasattr(impl, "set_sibling_providers"):
                impl.set_sibling_providers({key: value for key, value in siblings.items() if key != provider_id})


async def _execute_job(queue: JobQueue, record: JobRecord, impl: object, worker_id: str) -> None:
    dispatcher = get_dispatcher(record.api, record.method)
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat_loop(queue, record.job_id, worker_id, stop))
    try:
        kwargs = dispatcher.build_kwargs(record.payload)
        method = getattr(impl, record.method)
        with RequestProviderDataContext(user=_record_user(record)):
            result = await method(**kwargs)
        await queue.complete(record.job_id, worker_id, dispatcher.serialize_result(result))
        logger.debug("Job completed", job_id=record.job_id, api=record.api, method=record.method)
    except Exception as e:
        logger.warning("Job execution failed", job_id=record.job_id, error=str(e))
        await queue.fail(record.job_id, worker_id, f"Failed to execute job: {e}")
    finally:
        stop.set()
        await heartbeat
    for cleanup_record in await queue.claim_cleanup(worker_id, job_id=record.job_id, limit=1):
        await _cleanup_claimed_job(queue, cleanup_record, impl, worker_id)


def _record_user(record: JobRecord) -> User | None:
    user_data = record.authenticated_user
    if user_data is None:
        return None
    return User(
        user_data["principal"],
        user_data.get("attributes"),
        tenant_id=user_data.get("tenant_id"),
    )


async def _cleanup_claimed_job(queue: JobQueue, record: JobRecord, impl: object, worker_id: str) -> None:
    dispatcher = get_dispatcher(record.api, record.method)
    try:
        if dispatcher.cleanup is not None:
            with RequestProviderDataContext(user=_record_user(record)):
                await dispatcher.cleanup(impl, record.payload)
    except Exception as e:
        logger.warning("Failed to clean up terminal job resources", job_id=record.job_id, error=str(e))
        await queue.release_cleanup(record.job_id, worker_id)
    else:
        await queue.complete_cleanup(record.job_id, worker_id)


async def _cleanup_pending_jobs(
    queue: JobQueue,
    impls: dict[tuple[str, str], object],
    worker_id: str,
) -> None:
    for record in await queue.claim_cleanup(worker_id):
        impl = impls.get((record.api, record.provider_id))
        if impl is None:
            logger.warning(
                "Failed to clean up terminal job resources: provider is unavailable",
                job_id=record.job_id,
                api=record.api,
                provider_id=record.provider_id,
            )
            await queue.release_cleanup(record.job_id, worker_id)
            continue
        await _cleanup_claimed_job(queue, record, impl, worker_id)


async def _run_maintenance(
    queue: JobQueue,
    impls: dict[tuple[str, str], object],
    *,
    purge_expired: bool,
    worker_id: str = "maintenance",
) -> None:
    await queue.reclaim_stale()
    await _cleanup_pending_jobs(queue, impls, worker_id)
    if purge_expired:
        await queue.purge_expired()


async def _run_worker(config: WorkerConfig, stop_event: EventType) -> None:
    register_sqlstore_backends(config.backends)
    store = await get_system_sqlstore(SqlStoreReference(backend=config.jobs_backend, table_name=config.jobs_table))
    queue = JobQueue(store, config.jobs_table, config.lease_ttl_seconds)
    # The table already exists (the server created it), but SqlStore only learns a
    # table's schema by declaring it, and this is a fresh process with empty
    # metadata. Without this every query raises KeyError on the table name.
    await queue.initialize()

    impls: dict[tuple[str, str], object] = {}
    for descriptor in config.descriptors:
        impls[(descriptor.api, descriptor.provider_id)] = await _build_impl(descriptor)
    _wire_sibling_providers(impls)

    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    logger.info("Worker started", worker_id=worker_id, providers=[provider_id for _, provider_id in impls])

    loop = asyncio.get_running_loop()
    last_maintenance = 0.0
    while not stop_event.is_set():
        now = loop.time()
        purge_expired = now - last_maintenance >= config.maintenance_interval_seconds
        await _run_maintenance(queue, impls, purge_expired=purge_expired, worker_id=worker_id)
        if purge_expired:
            last_maintenance = now
        record = await queue.lease(worker_id)
        if record is None:
            await asyncio.sleep(config.poll_interval_seconds)
            continue
        impl = impls.get((record.api, record.provider_id))
        if impl is None:
            await queue.fail(record.job_id, worker_id, f"Worker has no impl for provider '{record.provider_id}'")
            continue
        await _execute_job(queue, record, impl, worker_id)

    logger.info("Worker stopping", worker_id=worker_id)


def _worker_main(config: WorkerConfig, stop_event: EventType) -> None:
    """Process entrypoint (must be module-level for the spawn start method)."""
    try:
        asyncio.run(_run_worker(config, stop_event))
    except Exception as e:
        # The traceback is formatted into the message because this is the only
        # record of a child's death: the parent sees an exit code, not the error.
        logger.error("Worker process crashed", error=str(e), traceback=traceback.format_exc())
        raise


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
        self._config: WorkerConfig | None = None
        self._ctx: SpawnContext | None = None
        self._supervisor: threading.Thread | None = None
        self._lock = threading.Lock()
        self._restart_failures: list[int] = []
        self._next_restart_at: list[float] = []
        self._started_at: list[float] = []
        self._started = False

    def set_backends(self, backends: dict[str, StorageBackendConfig]) -> None:
        self._backends = backends

    def register(self, descriptor: ProviderDescriptor) -> None:
        self._descriptors.append(descriptor)
        logger.debug("Registered worker provider", provider_id=descriptor.provider_id, api=descriptor.api)

    @property
    def registered_providers(self) -> set[tuple[str, str]]:
        return {(descriptor.api, descriptor.provider_id) for descriptor in self._descriptors}

    def _spawn_process(self) -> mp.process.BaseProcess:
        if self._ctx is None or self._config is None or self._stop_event is None:
            raise RuntimeError("Failed to spawn worker process: worker pool is not configured.")
        proc = self._ctx.Process(target=_worker_main, args=(self._config, self._stop_event), daemon=True)
        proc.start()
        return proc

    def _maintain_workers(self, now: float | None = None) -> None:
        if not self._started or self._stop_event is None or self._stop_event.is_set():
            return
        current_time = time.monotonic() if now is None else now
        with self._lock:
            for index, proc in enumerate(self._processes):
                if proc.is_alive():
                    if current_time - self._started_at[index] >= 60:
                        self._restart_failures[index] = 0
                        self._next_restart_at[index] = 0.0
                    continue
                if current_time < self._next_restart_at[index]:
                    continue
                proc.join(timeout=0)
                try:
                    replacement = self._spawn_process()
                except Exception as e:
                    self._restart_failures[index] += 1
                    delay = min(2 ** (self._restart_failures[index] - 1), 30)
                    self._next_restart_at[index] = current_time + delay
                    logger.error("Failed to restart worker process", worker_slot=index, error=str(e))
                    continue
                self._processes[index] = replacement
                self._restart_failures[index] += 1
                delay = min(2 ** (self._restart_failures[index] - 1), 30)
                self._next_restart_at[index] = current_time + delay
                self._started_at[index] = current_time
                logger.warning("Restarted worker process", worker_slot=index, restart_delay_seconds=delay)

    def _supervise(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(timeout=0.5):
            self._maintain_workers()

    @property
    def is_healthy(self) -> bool:
        if not self._descriptors and not self._started:
            return True
        if not self._started or self._stop_event is None or self._stop_event.is_set():
            return False
        with self._lock:
            return len(self._processes) == self.num_workers and all(proc.is_alive() for proc in self._processes)

    def start(self) -> None:
        if self._started or not self._descriptors:
            return
        self._config = WorkerConfig(
            backends=self._backends,
            jobs_backend=self.jobs_backend,
            jobs_table=self.jobs_table,
            lease_ttl_seconds=self.lease_ttl_seconds,
            descriptors=self._descriptors,
        )
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._stop_event = ctx.Event()
        started_at = time.monotonic()
        try:
            for _ in range(self.num_workers):
                self._processes.append(self._spawn_process())
                self._restart_failures.append(0)
                self._next_restart_at.append(0.0)
                self._started_at.append(started_at)
        except Exception:
            self._stop_event.set()
            for proc in self._processes:
                proc.join(timeout=0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            self._processes.clear()
            self._restart_failures.clear()
            self._next_restart_at.clear()
            self._started_at.clear()
            self._config = None
            self._ctx = None
            self._stop_event = None
            raise
        self._started = True
        self._supervisor = threading.Thread(target=self._supervise, name="ogx-job-worker-supervisor", daemon=True)
        self._supervisor.start()
        logger.info("Started worker pool", num_workers=self.num_workers, providers=len(self._descriptors))

    def shutdown(self, timeout: float = 10.0) -> None:
        if not self._started:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._supervisor is not None:
            self._supervisor.join(timeout=timeout)
            self._supervisor = None
        for proc in self._processes:
            proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        self._processes.clear()
        self._restart_failures.clear()
        self._next_restart_at.clear()
        self._started_at.clear()
        self._started = False
        logger.info("Worker pool shut down")
