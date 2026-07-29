# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Durable job queue backed by a SQL store.

The queue table is the single source of truth and doubles as the IPC channel
between the server (which enqueues and polls for results) and worker processes
(which lease, execute, and report). Leasing is an atomic guarded UPDATE: a row
can only be claimed when it is ``scheduled`` or its previous lease has expired,
which prevents two workers from running the same job.
"""

import time
import uuid

from ogx.log import get_logger
from ogx_api.common.job_types import JobStatus
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, SqlStore

from .models import JobRecord

logger = get_logger(name=__name__, category="core::jobs")

DEFAULT_LEASE_TTL_SECONDS = 60


class JobQueue:
    """A durable, SQL-backed work queue.

    Args:
        sql_store: The backing SQL store (shared DB across server and workers).
        table_name: Name of the queue table.
        lease_ttl_seconds: How long a lease is held before another worker may
            reclaim a stuck job.
    """

    def __init__(
        self, sql_store: SqlStore, table_name: str = "jobs", lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS
    ):
        self.sql_store = sql_store
        self.table_name = table_name
        self.lease_ttl_seconds = lease_ttl_seconds

    async def initialize(self) -> None:
        await self.sql_store.create_table(
            self.table_name,
            {
                "job_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "api": ColumnType.STRING,
                "provider_id": ColumnType.STRING,
                "method": ColumnType.STRING,
                "status": ColumnType.STRING,
                "payload": ColumnType.JSON,
                "result": ColumnType.JSON,
                "error": ColumnType.TEXT,
                "attempts": ColumnType.INTEGER,
                "max_attempts": ColumnType.INTEGER,
                "lease_owner": ColumnType.STRING,
                "lease_expires_at": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "updated_at": ColumnType.INTEGER,
            },
        )

    async def enqueue(
        self,
        api: str,
        provider_id: str,
        method: str,
        payload: dict,
        max_attempts: int = 1,
    ) -> JobRecord:
        """Insert a new ``scheduled`` job and return it."""
        now = int(time.time())
        record = JobRecord(
            job_id=f"job_{uuid.uuid4().hex[:24]}",
            api=api,
            provider_id=provider_id,
            method=method,
            status=JobStatus.scheduled,
            payload=payload,
            max_attempts=max_attempts,
            created_at=now,
            updated_at=now,
        )
        await self.sql_store.insert(self.table_name, self._to_row(record))
        logger.debug("Enqueued job", job_id=record.job_id, api=api, method=method)
        return record

    async def lease(self, worker_id: str) -> JobRecord | None:
        """Atomically claim one runnable job for ``worker_id``.

        A job is runnable when it is ``scheduled`` or it is ``in_progress`` with
        an expired lease (its previous worker died). Returns the claimed job, or
        ``None`` if nothing is available.
        """
        now = int(time.time())
        candidates = await self.sql_store.fetch_all(
            self.table_name,
            where_sql="status = :scheduled OR (status = :in_progress AND lease_expires_at < :now)",
            where_sql_params={
                "scheduled": JobStatus.scheduled.value,
                "in_progress": JobStatus.in_progress.value,
                "now": now,
            },
            order_by=[("created_at", "asc")],
            limit=20,
        )
        for row in candidates.data:
            record = self._from_row(row)
            if await self._try_claim(record, worker_id, now):
                claimed = await self.get(record.job_id)
                if claimed is not None and claimed.lease_owner == worker_id and claimed.status == JobStatus.in_progress:
                    return claimed
        return None

    async def _try_claim(self, record: JobRecord, worker_id: str, now: int) -> bool:
        """Issue the guarded claim UPDATE.

        The return value only reports whether the statement executed, not whether
        it matched a row: ``SqlStore.update()`` succeeds silently when the guard
        excludes every row, so a losing racer still gets ``True`` here. Exclusion
        is established by the caller's read-back in :meth:`lease`, which requires
        ``lease_owner == worker_id``; the guard on this UPDATE is what makes that
        read-back decisive, since only one racer's write can satisfy it.
        """
        try:
            await self.sql_store.update(
                self.table_name,
                data={
                    "status": JobStatus.in_progress.value,
                    "lease_owner": worker_id,
                    "lease_expires_at": now + self.lease_ttl_seconds,
                    "attempts": record.attempts + 1,
                    "updated_at": now,
                },
                where={"job_id": record.job_id},
                where_sql="status = :scheduled OR (status = :in_progress AND lease_expires_at < :now)",
                where_sql_params={
                    "scheduled": JobStatus.scheduled.value,
                    "in_progress": JobStatus.in_progress.value,
                    "now": now,
                },
            )
            return True
        except Exception as e:
            logger.warning("Failed to claim job", job_id=record.job_id, error=str(e))
            return False

    async def heartbeat(self, job_id: str, worker_id: str) -> None:
        """Extend the lease on an in-progress job still owned by ``worker_id``."""
        now = int(time.time())
        await self.sql_store.update(
            self.table_name,
            data={"lease_expires_at": now + self.lease_ttl_seconds, "updated_at": now},
            where={"job_id": job_id, "lease_owner": worker_id, "status": JobStatus.in_progress.value},
        )

    async def complete(self, job_id: str, worker_id: str, result: dict) -> None:
        """Mark a job completed. No-op if the job is no longer owned/in-progress (e.g. cancelled)."""
        now = int(time.time())
        await self.sql_store.update(
            self.table_name,
            data={"status": JobStatus.completed.value, "result": result, "error": None, "updated_at": now},
            where={"job_id": job_id, "lease_owner": worker_id, "status": JobStatus.in_progress.value},
        )

    async def fail(self, job_id: str, worker_id: str, error: str) -> None:
        """Mark a job failed, or re-queue it if attempts remain."""
        now = int(time.time())
        record = await self.get(job_id)
        if record is None:
            return
        if record.attempts < record.max_attempts:
            await self.sql_store.update(
                self.table_name,
                data={
                    "status": JobStatus.scheduled.value,
                    "lease_owner": None,
                    "lease_expires_at": 0,
                    "error": error,
                    "updated_at": now,
                },
                where={"job_id": job_id, "lease_owner": worker_id, "status": JobStatus.in_progress.value},
            )
            logger.info("Re-queued failed job", job_id=job_id, attempts=record.attempts, error=error)
        else:
            await self.sql_store.update(
                self.table_name,
                data={"status": JobStatus.failed.value, "error": error, "updated_at": now},
                where={"job_id": job_id, "lease_owner": worker_id, "status": JobStatus.in_progress.value},
            )
            logger.info("Job failed permanently", job_id=job_id, error=error)

    async def cancel(self, job_id: str) -> JobRecord | None:
        """Cancel a job. Scheduled jobs stop immediately; in-progress jobs are
        marked cancelled and the worker's eventual completion is discarded."""
        now = int(time.time())
        record = await self.get(job_id)
        if record is None or record.is_terminal:
            return record
        await self.sql_store.update(
            self.table_name,
            data={"status": JobStatus.cancelled.value, "updated_at": now},
            where={"job_id": job_id},
            where_sql="status = :scheduled OR status = :in_progress",
            where_sql_params={"scheduled": JobStatus.scheduled.value, "in_progress": JobStatus.in_progress.value},
        )
        return await self.get(job_id)

    async def reclaim_stale(self) -> int:
        """Return in-progress jobs with expired leases to the pool.

        Called on startup so work that was running when the server died is picked
        up again rather than lost. Jobs that have already used all their attempts
        are marked ``failed`` instead of rescheduled, matching the budget that
        :meth:`fail` enforces on the normal path. Returns the number of jobs
        reclaimed (rescheduled plus terminally failed).
        """
        now = int(time.time())
        stale = await self.sql_store.fetch_all(
            self.table_name,
            where_sql="status = :in_progress AND lease_expires_at < :now",
            where_sql_params={"in_progress": JobStatus.in_progress.value, "now": now},
        )
        rescheduled = 0
        exhausted = 0
        for row in stale.data:
            if row["attempts"] >= row["max_attempts"]:
                data = {
                    "status": JobStatus.failed.value,
                    "lease_owner": None,
                    "lease_expires_at": 0,
                    "error": "Failed to complete job: worker lease expired and no attempts remain",
                    "updated_at": now,
                }
                exhausted += 1
            else:
                data = {
                    "status": JobStatus.scheduled.value,
                    "lease_owner": None,
                    "lease_expires_at": 0,
                    "updated_at": now,
                }
                rescheduled += 1
            await self.sql_store.update(self.table_name, data=data, where={"job_id": row["job_id"]})
        if stale.data:
            logger.info("Reclaimed stale jobs on startup", rescheduled=rescheduled, failed=exhausted)
        return len(stale.data)

    async def get(self, job_id: str) -> JobRecord | None:
        row = await self.sql_store.fetch_one(self.table_name, where={"job_id": job_id})
        return self._from_row(row) if row is not None else None

    async def list(self, api: str | None = None, limit: int | None = None) -> list[JobRecord]:
        where = {"api": api} if api is not None else None
        results = await self.sql_store.fetch_all(
            self.table_name, where=where, order_by=[("created_at", "desc")], limit=limit
        )
        return [self._from_row(row) for row in results.data]

    @staticmethod
    def _to_row(record: JobRecord) -> dict:
        row = record.model_dump()
        row["status"] = record.status.value
        return row

    @staticmethod
    def _from_row(row: dict) -> JobRecord:
        return JobRecord.model_validate(row)
