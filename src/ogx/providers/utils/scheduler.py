# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import abc
import asyncio
import functools
import threading
from collections.abc import Callable, Coroutine, Iterable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from ogx.log import get_logger

logger = get_logger(name=__name__, category="providers::utils")


# TODO: revisit the list of possible statuses when defining a more coherent
# Jobs API for all API flows; e.g. do we need new vs scheduled?
class JobStatus(Enum):
    """Enumeration of possible job lifecycle states."""

    new = "new"
    scheduled = "scheduled"
    running = "running"
    failed = "failed"
    completed = "completed"


type JobID = str
type JobType = str


class JobArtifact(BaseModel):
    """Metadata and location information for an artifact produced by a job."""

    type: JobType
    name: str
    # TODO: uri should be a reference to /files API; revisit when /files is implemented
    uri: str | None = None
    metadata: dict[str, Any]


JobHandler = Callable[
    [Callable[[str], None], Callable[[JobStatus], None], Callable[[JobArtifact], None]], Coroutine[Any, Any, None]
]


type LogMessage = tuple[datetime, str]


_COMPLETED_STATUSES = {JobStatus.completed, JobStatus.failed}


class Job:
    """Represents a scheduled unit of work with status tracking, logs, and artifacts."""

    def __init__(self, job_type: JobType, job_id: JobID, handler: JobHandler):
        super().__init__()
        self.id = job_id
        self._type = job_type
        self._handler = handler
        self._artifacts: list[JobArtifact] = []
        self._logs: list[LogMessage] = []
        self._state_transitions: list[tuple[datetime, JobStatus]] = [(datetime.now(UTC), JobStatus.new)]

    @property
    def handler(self) -> JobHandler:
        return self._handler

    @property
    def status(self) -> JobStatus:
        return self._state_transitions[-1][1]

    @status.setter
    def status(self, status: JobStatus):
        if status in _COMPLETED_STATUSES and self.status in _COMPLETED_STATUSES:
            raise ValueError(f"Job is already in a completed state ({self.status})")
        if self.status == status:
            return
        self._state_transitions.append((datetime.now(UTC), status))

    @property
    def artifacts(self) -> list[JobArtifact]:
        return self._artifacts

    def register_artifact(self, artifact: JobArtifact) -> None:
        self._artifacts.append(artifact)

    def _find_state_transition_date(self, status: Iterable[JobStatus]) -> datetime | None:
        for date, s in reversed(self._state_transitions):
            if s in status:
                return date
        return None

    @property
    def scheduled_at(self) -> datetime | None:
        return self._find_state_transition_date([JobStatus.scheduled])

    @property
    def started_at(self) -> datetime | None:
        return self._find_state_transition_date([JobStatus.running])

    @property
    def completed_at(self) -> datetime | None:
        return self._find_state_transition_date(_COMPLETED_STATUSES)

    @property
    def logs(self) -> list[LogMessage]:
        return self._logs[:]

    def append_log(self, message: LogMessage) -> None:
        self._logs.append(message)

    # TODO: implement
    def cancel(self) -> None:
        raise NotImplementedError


class _SchedulerBackend(abc.ABC):
    @abc.abstractmethod
    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        raise NotImplementedError


class _NaiveSchedulerBackend(_SchedulerBackend):
    def __init__(self, timeout: int = 5):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        # There may be performance implications of using threads due to Python
        # GIL; may need to measure if it's a real problem though
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

        # TODO: When stopping the loop, give tasks a chance to finish
        # TODO: should we explicitly inform jobs of pending stoppage?

        # cancel all tasks
        for task in asyncio.all_tasks(self._loop):
            if not task.done():
                task.cancel()

        self._loop.close()

    async def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    # TODO: decouple scheduling and running the job
    def schedule(
        self,
        job: Job,
        on_log_message_cb: Callable[[str], None],
        on_status_change_cb: Callable[[JobStatus], None],
        on_artifact_collected_cb: Callable[[JobArtifact], None],
    ) -> None:
        async def do():
            try:
                job.status = JobStatus.running
                await job.handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb)
            except Exception as e:
                on_log_message_cb(str(e))
                job.status = JobStatus.failed
                logger.exception("Job failed", job_id=job.id)

        asyncio.run_coroutine_threadsafe(do(), self._loop)

    def on_log_message_cb(self, job: Job, message: LogMessage) -> None:
        pass

    def on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        pass

    def on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        pass


_BACKENDS = {
    "naive": _NaiveSchedulerBackend,
}


def _get_backend_impl(backend: str) -> _SchedulerBackend:
    try:
        return _BACKENDS[backend]()
    except KeyError as e:
        raise ValueError(f"Unknown backend {backend}") from e


class Scheduler:
    """Manages job scheduling, execution, and lifecycle using a pluggable backend."""

    def __init__(self, backend: str = "naive"):
        # TODO: if server crashes, job states are lost; we need to persist jobs on disc
        self._jobs: dict[JobID, Job] = {}
        self._backend = _get_backend_impl(backend)

    def _on_log_message_cb(self, job: Job, message: str) -> None:
        msg = (datetime.now(UTC), message)
        # At least for the time being, until there's a better way to expose
        # logs to users, log messages on console
        logger.info(message, job_id=job.id)
        job.append_log(msg)
        self._backend.on_log_message_cb(job, msg)

    def _on_status_change_cb(self, job: Job, status: JobStatus) -> None:
        job.status = status
        self._backend.on_status_change_cb(job, status)

    def _on_artifact_collected_cb(self, job: Job, artifact: JobArtifact) -> None:
        job.register_artifact(artifact)
        self._backend.on_artifact_collected_cb(job, artifact)

    def schedule(self, type_: JobType, job_id: JobID, handler: JobHandler) -> JobID:
        job = Job(type_, job_id, handler)
        if job.id in self._jobs:
            raise ValueError(f"Job {job.id} already exists")

        self._jobs[job.id] = job
        job.status = JobStatus.scheduled
        self._backend.schedule(
            job,
            functools.partial(self._on_log_message_cb, job),
            functools.partial(self._on_status_change_cb, job),
            functools.partial(self._on_artifact_collected_cb, job),
        )

        return job.id

    def cancel(self, job_id: JobID) -> None:
        self.get_job(job_id).cancel()

    def get_job(self, job_id: JobID) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as e:
            raise ValueError(f"Job {job_id} not found") from e

    def get_jobs(self, type_: JobType | None = None) -> list[Job]:
        jobs = list(self._jobs.values())
        if type_:
            jobs = [job for job in jobs if job._type == type_]
        return jobs

    async def shutdown(self):
        # TODO: also cancel jobs once implemented
        await self._backend.shutdown()
