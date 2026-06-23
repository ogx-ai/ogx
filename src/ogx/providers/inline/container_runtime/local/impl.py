# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Local Docker/Podman container runtime.

Implements the internal :class:`ogx_api.containers.ContainerRuntime` protocol on
top of a local container engine via the docker-py SDK. Podman is supported
through its Docker-compatible API socket, selected with ``engine: podman``.

The engine holds the live container; container and file *metadata* live in a SQL
store so listing, pagination, expiry, and ``last_active_at`` tracking survive
across engine restarts. The docker client is created lazily so the server can
start without a reachable engine — only actual container operations require one.
"""

import asyncio
import io
import json
import tarfile
import time
import uuid
from typing import Any

from fastapi import Response, UploadFile

from ogx.core.datatypes import AccessRule
from ogx.core.id_generation import generate_object_id
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore, authorized_sqlstore
from ogx.log import get_logger
from ogx_api import (
    Container,
    ContainerCreateRequest,
    ContainerDeleteResponse,
    ContainerExpiresAfter,
    ContainerFile,
    ContainerFileDeleteResponse,
    ContainerFileNotFoundError,
    ContainerFileSource,
    ContainerMemoryLimit,
    ContainerNotFoundError,
    ContainerStatus,
    DeleteContainerFileRequest,
    DeleteContainerRequest,
    ExecuteShellRequest,
    GetContainerFileContentRequest,
    GetContainerFileRequest,
    GetContainerRequest,
    ListContainerFilesRequest,
    ListContainerFilesResponse,
    ListContainersRequest,
    ListContainersResponse,
    MountSkillsRequest,
    NetworkPolicyMode,
    Order,
    ShellCallOutput,
    ShellOutcomeFailure,
    ShellOutcomeSuccess,
    ShellOutcomeTimeout,
    UploadContainerFileRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

from .config import LocalContainerRuntimeConfig

logger = get_logger(name=__name__, category="container_runtime")

_CONTAINERS_TABLE = "container_runtime_containers"
_FILES_TABLE = "container_runtime_files"

_DATA_DIR = "/mnt/data"
_SKILLS_DIR = "/mnt/skills"
_CONTAINER_LABEL = "ogx.container_id"

_MEMORY_LIMIT_BYTES: dict[ContainerMemoryLimit, int] = {
    ContainerMemoryLimit.GB_1: 1 * 1024**3,
    ContainerMemoryLimit.GB_4: 4 * 1024**3,
    ContainerMemoryLimit.GB_16: 16 * 1024**3,
    ContainerMemoryLimit.GB_64: 64 * 1024**3,
}


def _now() -> int:
    return int(time.time())


def _tar_bytes(arcname: str, data: bytes) -> bytes:
    """Build an in-memory tar archive holding a single file, for ``put_archive``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mtime = _now()
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _first_file_from_tar(tar_bytes: bytes) -> bytes:
    """Extract the bytes of the first regular file from a tar stream."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is not None:
                    return extracted.read()
    return b""


class LocalContainerRuntimeImpl:
    """ContainerRuntime backed by a local Docker or Podman engine."""

    def __init__(self, config: LocalContainerRuntimeConfig, policy: list[AccessRule]) -> None:
        self.config = config
        self.policy = policy
        self.sql_store: AuthorizedSqlStore | None = None
        self._docker_client: Any | None = None
        self._expiration_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        # Note: no engine socket access here — the docker client connects lazily so the
        # server can boot without Docker/Podman present (e.g. in CI).
        self.sql_store = await authorized_sqlstore(self.config.metadata_store, self.policy)
        await self.sql_store.create_table(
            _CONTAINERS_TABLE,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "backend_id": ColumnType.STRING,
                "name": ColumnType.STRING,
                "image": ColumnType.STRING,
                "memory_limit": ColumnType.STRING,
                "status": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "last_active_at": ColumnType.INTEGER,
                "expires_after_minutes": ColumnType.INTEGER,
                "network_policy": ColumnType.STRING,
            },
        )
        await self.sql_store.create_table(
            _FILES_TABLE,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "container_id": ColumnType.STRING,
                "path": ColumnType.STRING,
                "bytes": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
                "source": ColumnType.STRING,
            },
        )
        self._expiration_task = asyncio.create_task(self._expiration_loop())

    async def shutdown(self) -> None:
        if self._expiration_task is not None:
            self._expiration_task.cancel()
            try:
                await self._expiration_task
            except asyncio.CancelledError:
                pass
            self._expiration_task = None
        if self._docker_client is not None:
            await asyncio.to_thread(self._docker_client.close)
            self._docker_client = None

    # --- engine plumbing -------------------------------------------------

    def _client(self) -> Any:
        """Return a cached docker-py client, connecting to the engine on first use."""
        if self._docker_client is None:
            import docker  # type: ignore[import-untyped]

            self._docker_client = docker.DockerClient(base_url=self.config.resolved_socket_url())
        return self._docker_client

    async def _backend_container(self, backend_id: str) -> Any:
        client = self._client()
        return await asyncio.to_thread(client.containers.get, backend_id)

    # --- metadata helpers ------------------------------------------------

    def _store(self) -> AuthorizedSqlStore:
        if self.sql_store is None:
            raise RuntimeError("Failed to use container runtime: provider not initialized")
        return self.sql_store

    @staticmethod
    def _row_to_container(row: dict[str, Any]) -> Container:
        network_policy = json.loads(row["network_policy"]) if row.get("network_policy") else None
        expires_after = None
        if row.get("expires_after_minutes") is not None:
            expires_after = ContainerExpiresAfter(minutes=row["expires_after_minutes"])
        return Container(
            id=row["id"],
            created_at=row["created_at"],
            status=ContainerStatus(row["status"]),
            last_active_at=row["last_active_at"],
            name=row.get("name"),
            image=row.get("image"),
            memory_limit=ContainerMemoryLimit(row.get("memory_limit") or ContainerMemoryLimit.GB_1.value),
            expires_after=expires_after,
            network_policy=network_policy,
        )

    @staticmethod
    def _row_to_file(row: dict[str, Any]) -> ContainerFile:
        return ContainerFile(
            id=row["id"],
            container_id=row["container_id"],
            created_at=row["created_at"],
            bytes=row["bytes"],
            path=row["path"],
            source=ContainerFileSource(row["source"]),
        )

    async def _get_container_row(self, container_id: str) -> dict[str, Any]:
        row = await self._store().fetch_one(_CONTAINERS_TABLE, where={"id": container_id})
        if not row:
            raise ContainerNotFoundError(container_id)
        return row

    async def _touch(self, container_id: str) -> None:
        """Refresh ``last_active_at`` after an operation against the container."""
        await self._store().update(_CONTAINERS_TABLE, {"last_active_at": _now()}, where={"id": container_id})

    # --- lifecycle -------------------------------------------------------

    async def create_container(self, request: ContainerCreateRequest) -> Container:
        store = self._store()
        active = await store.fetch_all(_CONTAINERS_TABLE, where={"status": ContainerStatus.ACTIVE.value})
        if len(active.data) >= self.config.max_containers:
            raise RuntimeError(f"Failed to create container: max_containers ({self.config.max_containers}) reached")

        container_id = generate_object_id("container", lambda: f"container-{uuid.uuid4().hex}")
        image = request.image or self.config.default_image
        network_disabled = True
        if request.network_policy is not None and request.network_policy.mode != NetworkPolicyMode.DENY:
            network_disabled = False

        client = self._client()

        def _run() -> str:
            container = client.containers.run(
                image,
                command=["sh", "-c", f"mkdir -p {_DATA_DIR} {_SKILLS_DIR} && exec sleep infinity"],
                detach=True,
                network_disabled=network_disabled,
                mem_limit=_MEMORY_LIMIT_BYTES[request.memory_limit],
                labels={_CONTAINER_LABEL: container_id},
                working_dir=_DATA_DIR,
            )
            return str(container.id)

        backend_id = await asyncio.to_thread(_run)

        now = _now()
        network_policy_json = request.network_policy.model_dump_json() if request.network_policy is not None else None
        await store.insert(
            _CONTAINERS_TABLE,
            {
                "id": container_id,
                "backend_id": backend_id,
                "name": request.name,
                "image": image,
                "memory_limit": request.memory_limit.value,
                "status": ContainerStatus.ACTIVE.value,
                "created_at": now,
                "last_active_at": now,
                "expires_after_minutes": request.expires_after.minutes if request.expires_after else None,
                "network_policy": network_policy_json,
            },
        )
        logger.info("Created container", container_id=container_id, image=image, engine=self.config.engine)
        return await self.get_container(GetContainerRequest(container_id=container_id))

    async def get_container(self, request: GetContainerRequest) -> Container:
        row = await self._get_container_row(request.container_id)
        return self._row_to_container(row)

    async def list_containers(self, request: ListContainersRequest) -> ListContainersResponse:
        order = request.order or Order.desc
        limit = request.limit or 20
        cursor = ("id", request.after) if request.after else None
        page = await self._store().fetch_all(
            _CONTAINERS_TABLE,
            limit=limit,
            order_by=[("created_at", order.value)],
            cursor=cursor,
        )
        containers = [self._row_to_container(r) for r in page.data]
        return ListContainersResponse(
            data=containers,
            first_id=containers[0].id if containers else None,
            last_id=containers[-1].id if containers else None,
            has_more=page.has_more,
        )

    async def delete_container(self, request: DeleteContainerRequest) -> ContainerDeleteResponse:
        row = await self._get_container_row(request.container_id)
        await self._remove_backend(row["backend_id"])
        await self._store().delete(_CONTAINERS_TABLE, where={"id": request.container_id})
        await self._store().delete(_FILES_TABLE, where={"container_id": request.container_id})
        return ContainerDeleteResponse(id=request.container_id, deleted=True)

    async def _remove_backend(self, backend_id: str | None) -> None:
        if not backend_id:
            return
        try:
            container = await self._backend_container(backend_id)
            await asyncio.to_thread(container.remove, force=True)
        except Exception as err:  # noqa: BLE001 - best-effort cleanup; engine state may already be gone
            logger.warning("Failed to remove backend container", backend_id=backend_id, error=str(err))

    # --- file management -------------------------------------------------

    async def upload_file(self, request: UploadContainerFileRequest, file: UploadFile) -> ContainerFile:
        row = await self._get_container_row(request.container_id)
        data = await file.read()
        filename = file.filename or "file"
        path = f"{_DATA_DIR}/{filename}"

        container = await self._backend_container(row["backend_id"])
        await asyncio.to_thread(container.put_archive, _DATA_DIR, _tar_bytes(filename, data))

        file_id = generate_object_id("container_file", lambda: f"container_file-{uuid.uuid4().hex}")
        now = _now()
        await self._store().insert(
            _FILES_TABLE,
            {
                "id": file_id,
                "container_id": request.container_id,
                "path": path,
                "bytes": len(data),
                "created_at": now,
                "source": ContainerFileSource.USER.value,
            },
        )
        await self._touch(request.container_id)
        return ContainerFile(
            id=file_id,
            container_id=request.container_id,
            created_at=now,
            bytes=len(data),
            path=path,
            source=ContainerFileSource.USER,
        )

    async def list_files(self, request: ListContainerFilesRequest) -> ListContainerFilesResponse:
        await self._get_container_row(request.container_id)
        order = request.order or Order.desc
        limit = request.limit or 20
        cursor = ("id", request.after) if request.after else None
        page = await self._store().fetch_all(
            _FILES_TABLE,
            where={"container_id": request.container_id},
            limit=limit,
            order_by=[("created_at", order.value)],
            cursor=cursor,
        )
        files = [self._row_to_file(r) for r in page.data]
        return ListContainerFilesResponse(
            data=files,
            first_id=files[0].id if files else None,
            last_id=files[-1].id if files else None,
            has_more=page.has_more,
        )

    async def _get_file_row(self, container_id: str, file_id: str) -> dict[str, Any]:
        row = await self._store().fetch_one(_FILES_TABLE, where={"id": file_id, "container_id": container_id})
        if not row:
            raise ContainerFileNotFoundError(file_id, container_id)
        return row

    async def get_file(self, request: GetContainerFileRequest) -> ContainerFile:
        await self._get_container_row(request.container_id)
        row = await self._get_file_row(request.container_id, request.file_id)
        return self._row_to_file(row)

    async def get_file_content(self, request: GetContainerFileContentRequest) -> Response:
        container_row = await self._get_container_row(request.container_id)
        file_row = await self._get_file_row(request.container_id, request.file_id)
        container = await self._backend_container(container_row["backend_id"])

        def _read() -> bytes:
            stream, _stat = container.get_archive(file_row["path"])
            return _first_file_from_tar(b"".join(stream))

        content = await asyncio.to_thread(_read)
        filename = file_row["path"].rsplit("/", 1)[-1]
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    async def delete_file(self, request: DeleteContainerFileRequest) -> ContainerFileDeleteResponse:
        container_row = await self._get_container_row(request.container_id)
        file_row = await self._get_file_row(request.container_id, request.file_id)
        container = await self._backend_container(container_row["backend_id"])
        await asyncio.to_thread(container.exec_run, ["rm", "-f", file_row["path"]])
        await self._store().delete(_FILES_TABLE, where={"id": request.file_id})
        await self._touch(request.container_id)
        return ContainerFileDeleteResponse(id=request.file_id, deleted=True)

    # --- execution -------------------------------------------------------

    async def execute_shell(self, request: ExecuteShellRequest) -> ShellCallOutput:
        row = await self._get_container_row(request.container_id)
        container = await self._backend_container(row["backend_id"])

        started = time.monotonic()

        def _exec() -> tuple[int, bytes | None, bytes | None]:
            result = container.exec_run(cmd=request.command, demux=True)
            stdout, stderr = result.output if isinstance(result.output, tuple) else (result.output, None)
            return result.exit_code, stdout, stderr

        timed_out = False
        exit_code: int | None = None
        stdout_b: bytes | None = None
        stderr_b: bytes | None = None
        try:
            if request.timeout_seconds is not None:
                exit_code, stdout_b, stderr_b = await asyncio.wait_for(
                    asyncio.to_thread(_exec), timeout=request.timeout_seconds
                )
            else:
                exit_code, stdout_b, stderr_b = await asyncio.to_thread(_exec)
        except TimeoutError:
            timed_out = True

        elapsed = time.monotonic() - started
        await self._touch(request.container_id)

        if timed_out:
            outcome: Any = ShellOutcomeTimeout(elapsed_seconds=elapsed)
        elif exit_code == 0:
            outcome = ShellOutcomeSuccess()
        else:
            outcome = ShellOutcomeFailure(exit_code=exit_code or 1)

        return ShellCallOutput(
            stdout=(stdout_b or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_b or b"").decode("utf-8", errors="replace"),
            outcome=outcome,
            duration_ms=int(elapsed * 1000),
            container_id=request.container_id,
        )

    # --- skill mounting --------------------------------------------------

    async def mount_skills(self, request: MountSkillsRequest) -> None:
        row = await self._get_container_row(request.container_id)
        container = await self._backend_container(row["backend_id"])
        for skill_name, zip_bytes in request.skill_bundles:
            target = f"{_SKILLS_DIR}/{skill_name}"
            tar_bytes = _zip_to_tar(zip_bytes)
            await asyncio.to_thread(container.exec_run, ["mkdir", "-p", target])
            await asyncio.to_thread(container.put_archive, target, tar_bytes)
        await self._touch(request.container_id)

    # --- expiration ------------------------------------------------------

    async def _expiration_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.expiration_poll_seconds)
                await self._reap_expired()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - keep the reaper alive across transient errors
                logger.warning("Container expiration sweep failed", error=str(err))

    async def _reap_expired(self) -> None:
        now = _now()
        page = await self._store().fetch_all(_CONTAINERS_TABLE, where={"status": ContainerStatus.ACTIVE.value})
        for row in page.data:
            minutes = row.get("expires_after_minutes")
            if minutes is None:
                continue
            if row["last_active_at"] + minutes * 60 < now:
                logger.info("Reaping expired container", container_id=row["id"])
                await self._remove_backend(row.get("backend_id"))
                await self._store().update(
                    _CONTAINERS_TABLE, {"status": ContainerStatus.EXPIRED.value}, where={"id": row["id"]}
                )


def _zip_to_tar(zip_bytes: bytes) -> bytes:
    """Convert a zip archive into a tar archive for ``put_archive`` extraction."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf, tarfile.open(fileobj=buf, mode="w") as tar:
        for name in zf.namelist():
            data = zf.read(name)
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = _now()
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()
