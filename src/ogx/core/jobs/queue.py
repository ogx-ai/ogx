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

from ogx.core.datatypes import User
from ogx.log import get_logger
from ogx_api.common.job_types import JobStatus
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, SqlStore

from .models import JobRecord

logger = get_logger(name=__name__, category="core::jobs")

DEFAULT_LEASE_TTL_SECONDS = 60
DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60


class JobQueue:
    """A durable, SQL-backed work queue.

    Args:
        sql_store: The backing SQL store (shared DB across server and workers).
        table_name: Name of the queue table.
        lease_ttl_seconds: How long a lease is held before another worker may
            reclaim a stuck job.
    """

    def __init__(
        self,
        sql_store: SqlStore,
        table_name: str = "jobs",
        lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    ):
        self.sql_store = sql_store
        self.table_name = table_name
        self.lease_ttl_seconds = lease_ttl_seconds
        self.retention_seconds = retention_seconds

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
                "owner_principal": ColumnType.STRING,
                "owner_tenant_id": ColumnType.STRING,
                "authenticated_user": ColumnType.JSON,
                "result": ColumnType.JSON,
                "error": ColumnType.TEXT,
                "attempts": ColumnType.INTEGER,
                "max_attempts": ColumnType.INTEGER,
                "lease_owner": ColumnType.STRING,
                "lease_expires_at": ColumnType.INTEGER,
                "cleanup_owner": ColumnType.STRING,
                "cleanup_lease_expires_at": ColumnType.INTEGER,
                "cleaned_at": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "updated_at": ColumnType.INTEGER,
            },
        )
        await self.sql_store.add_column_if_not_exists(self.table_name, "owner_principal", ColumnType.STRING)
        await self.sql_store.add_column_if_not_exists(self.table_name, "owner_tenant_id", ColumnType.STRING)
        await self.sql_store.add_column_if_not_exists(self.table_name, "authenticated_user", ColumnType.JSON)
        await self.sql_store.add_column_if_not_exists(self.table_name, "cleanup_owner", ColumnType.STRING)
        await self.sql_store.add_column_if_not_exists(self.table_name, "cleanup_lease_expires_at", ColumnType.INTEGER)
        await self.sql_store.add_column_if_not_exists(self.table_name, "cleaned_at", ColumnType.INTEGER)
        await self.sql_store.create_index(
            f"idx_{self.table_name}_scheduled",
            self.table_name,
            ["status", "created_at"],
        )
        await self.sql_store.create_index(
            f"idx_{self.table_name}_leases",
            self.table_name,
            ["status", "lease_expires_at"],
        )
        await self.sql_store.create_index(
            f"idx_{self.table_name}_retention",
            self.table_name,
            ["status", "updated_at"],
        )
        await self.sql_store.create_index(
            f"idx_{self.table_name}_cleanup",
            self.table_name,
            ["cleaned_at", "cleanup_lease_expires_at", "updated_at"],
        )
        await self.sql_store.create_index(
            f"idx_{self.table_name}_scope_created",
            self.table_name,
            ["api", "provider_id", "owner_tenant_id", "owner_principal", "created_at", "job_id"],
        )
        await self.purge_expired()

    async def enqueue(
        self,
        api: str,
        provider_id: str,
        method: str,
        payload: dict,
        max_attempts: int = 1,
        authenticated_user: User | None = None,
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
            owner_principal=authenticated_user.principal if authenticated_user is not None else None,
            owner_tenant_id=authenticated_user.tenant_id if authenticated_user is not None else None,
            authenticated_user=authenticated_user.model_dump(mode="json") if authenticated_user is not None else None,
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
            where_sql="status = :scheduled AND attempts < max_attempts",
            where_sql_params={
                "scheduled": JobStatus.scheduled.value,
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
                where_sql="status = :scheduled AND attempts < max_attempts",
                where_sql_params={
                    "scheduled": JobStatus.scheduled.value,
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

        Called by periodic worker maintenance so work interrupted by a crashed
        worker is picked up again rather than lost. Jobs that have used all their attempts
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
            await self.sql_store.update(
                self.table_name,
                data=data,
                where={"job_id": row["job_id"]},
                where_sql="status = :in_progress AND lease_expires_at < :now",
                where_sql_params={"in_progress": JobStatus.in_progress.value, "now": now},
            )
        if stale.data:
            logger.info("Reclaimed stale jobs", rescheduled=rescheduled, failed=exhausted)
        return len(stale.data)

    async def claim_cleanup(
        self,
        worker_id: str,
        job_id: str | None = None,
        limit: int = 20,
    ) -> list[JobRecord]:
        """Claim terminal jobs whose resource cleanup has not completed."""
        now = int(time.time())
        where = {"job_id": job_id} if job_id is not None else None
        cleanup_filter = (
            "status IN (:completed, :failed, :cancelled) AND cleaned_at IS NULL "
            "AND (cleanup_owner IS NULL OR cleanup_lease_expires_at < :now)"
        )
        params = {
            "completed": JobStatus.completed.value,
            "failed": JobStatus.failed.value,
            "cancelled": JobStatus.cancelled.value,
            "now": now,
        }
        candidates = await self.sql_store.fetch_all(
            self.table_name,
            where=where,
            where_sql=cleanup_filter,
            where_sql_params=params,
            order_by=[("updated_at", "asc")],
            limit=limit,
        )
        claimed: list[JobRecord] = []
        for row in candidates.data:
            await self.sql_store.update(
                self.table_name,
                data={
                    "cleanup_owner": worker_id,
                    "cleanup_lease_expires_at": now + self.lease_ttl_seconds,
                },
                where={"job_id": row["job_id"]},
                where_sql=cleanup_filter,
                where_sql_params=params,
            )
            record = await self.get(row["job_id"])
            if record is not None and record.cleanup_owner == worker_id and record.cleaned_at is None:
                claimed.append(record)
        return claimed

    async def complete_cleanup(self, job_id: str, worker_id: str) -> None:
        """Mark a claimed terminal job's resource cleanup complete."""
        await self.sql_store.update(
            self.table_name,
            data={
                "cleaned_at": int(time.time()),
                "cleanup_owner": None,
                "cleanup_lease_expires_at": 0,
            },
            where={"job_id": job_id, "cleanup_owner": worker_id},
            where_sql="cleaned_at IS NULL",
        )

    async def release_cleanup(self, job_id: str, worker_id: str) -> None:
        """Release failed cleanup work so another maintenance pass can retry it."""
        await self.sql_store.update(
            self.table_name,
            data={"cleanup_owner": None, "cleanup_lease_expires_at": 0},
            where={"job_id": job_id, "cleanup_owner": worker_id},
            where_sql="cleaned_at IS NULL",
        )

    async def get(self, job_id: str) -> JobRecord | None:
        row = await self.sql_store.fetch_one(self.table_name, where={"job_id": job_id})
        return self._from_row(row) if row is not None else None

    @staticmethod
    def _scope(api: str, provider_id: str, authenticated_user: User | None) -> dict[str, str | None]:
        return {
            "api": api,
            "provider_id": provider_id,
            "owner_principal": authenticated_user.principal if authenticated_user is not None else None,
            "owner_tenant_id": authenticated_user.tenant_id if authenticated_user is not None else None,
        }

    async def get_scoped(
        self,
        job_id: str,
        api: str,
        provider_id: str,
        authenticated_user: User | None,
    ) -> JobRecord | None:
        row = await self.sql_store.fetch_one(
            self.table_name,
            where={"job_id": job_id, **self._scope(api, provider_id, authenticated_user)},
        )
        return self._from_row(row) if row is not None else None

    async def cancel_scoped(
        self,
        job_id: str,
        api: str,
        provider_id: str,
        authenticated_user: User | None,
    ) -> JobRecord | None:
        record = await self.get_scoped(job_id, api, provider_id, authenticated_user)
        if record is None or record.is_terminal:
            return record
        now = int(time.time())
        await self.sql_store.update(
            self.table_name,
            data={"status": JobStatus.cancelled.value, "updated_at": now},
            where={"job_id": job_id, **self._scope(api, provider_id, authenticated_user)},
            where_sql="status = :scheduled OR status = :in_progress",
            where_sql_params={"scheduled": JobStatus.scheduled.value, "in_progress": JobStatus.in_progress.value},
        )
        return await self.get_scoped(job_id, api, provider_id, authenticated_user)

    async def list_scoped(
        self,
        api: str,
        provider_id: str,
        authenticated_user: User | None,
        after: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[JobRecord], bool]:
        where_sql = None
        where_sql_params = None
        if after is not None:
            cursor = await self.get_scoped(after, api, provider_id, authenticated_user)
            if cursor is None:
                raise ValueError(f"Failed to find job pagination cursor '{after}'.")
            where_sql = (
                "(created_at < :cursor_created_at OR (created_at = :cursor_created_at AND job_id < :cursor_job_id))"
            )
            where_sql_params = {"cursor_created_at": cursor.created_at, "cursor_job_id": cursor.job_id}
        results = await self.sql_store.fetch_all(
            self.table_name,
            where=self._scope(api, provider_id, authenticated_user),
            where_sql=where_sql,
            where_sql_params=where_sql_params,
            order_by=[("created_at", "desc"), ("job_id", "desc")],
            limit=limit,
        )
        return [self._from_row(row) for row in results.data], results.has_more

    async def purge_expired(self, now: int | None = None) -> None:
        """Delete terminal job records after the configured retention window."""
        cutoff = (int(time.time()) if now is None else now) - self.retention_seconds
        await self.sql_store.delete(
            self.table_name,
            where={"updated_at": {"<": cutoff}},
            where_sql="status IN (:completed, :failed, :cancelled) AND cleaned_at IS NOT NULL",
            where_sql_params={
                "completed": JobStatus.completed.value,
                "failed": JobStatus.failed.value,
                "cancelled": JobStatus.cancelled.value,
            },
        )

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
