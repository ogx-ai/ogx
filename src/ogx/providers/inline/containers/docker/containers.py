# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import time
import uuid

try:
    import docker
except ImportError:
    docker = None  # type: ignore[assignment]

from ogx.core.datatypes import AccessRule
from ogx.core.id_generation import generate_object_id
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore, authorized_sqlstore
from ogx.log import get_logger
from ogx_api import ResourceNotFoundError
from ogx_api.containers import (
    Container,
    ContainerExpiresAfter,
    Containers,
    CreateContainerRequest,
    DeleteContainerRequest,
    DeleteContainerResponse,
    ExecInContainerRequest,
    ExecInContainerResponse,
    ListContainersRequest,
    ListContainersResponse,
    NetworkPolicyAllowlist,
    NetworkPolicyDisabled,
    RetrieveContainerRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

from .config import DockerContainersImplConfig

logger = get_logger(name=__name__, category="containers")


class DockerContainersImpl(Containers):
    """Containers provider that manages sandboxed environments via Docker."""

    def __init__(self, config: DockerContainersImplConfig, policy: list[AccessRule]) -> None:
        self.config = config
        self.policy = policy
        self.sql_store: AuthorizedSqlStore | None = None
        self.docker_client = None

    async def initialize(self) -> None:
        if docker is not None:
            try:
                self.docker_client = await asyncio.to_thread(docker.from_env)  # type: ignore[func-returns-value]
            except Exception as e:
                logger.warning("Failed to connect to Docker daemon", error=str(e))

        self.sql_store = await authorized_sqlstore(self.config.metadata_store, self.policy)
        await self.sql_store.create_table(
            "containers",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "name": ColumnType.STRING,
                "status": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "last_active_at": ColumnType.INTEGER,
                "expires_after_minutes": ColumnType.INTEGER,
                "memory_limit": ColumnType.STRING,
                "network_policy_json": ColumnType.STRING,
                "docker_container_id": ColumnType.STRING,
            },
        )

    async def shutdown(self) -> None:
        if self.docker_client:
            await asyncio.to_thread(self.docker_client.close)

    def _row_to_container(self, row: dict) -> Container:
        network_policy: NetworkPolicyAllowlist | NetworkPolicyDisabled | None = None
        policy_json = row.get("network_policy_json")
        if policy_json:
            policy_data = json.loads(policy_json)
            if policy_data.get("type") == "allowlist":
                network_policy = NetworkPolicyAllowlist(**policy_data)
            elif policy_data.get("type") == "disabled":
                network_policy = NetworkPolicyDisabled()

        expires_after = None
        if row.get("expires_after_minutes"):
            expires_after = ContainerExpiresAfter(minutes=row["expires_after_minutes"])

        return Container(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            created_at=row["created_at"],
            last_active_at=row.get("last_active_at"),
            expires_after=expires_after,
            memory_limit=row.get("memory_limit"),
            network_policy=network_policy,
        )

    async def create_container(self, request: CreateContainerRequest) -> Container:
        if not self.sql_store:
            raise RuntimeError("Failed to create container: provider not initialized")

        container_id = generate_object_id("cntr", lambda: f"cntr-{uuid.uuid4().hex}")
        now = int(time.time())
        memory_limit = request.memory_limit or self.config.default_memory_limit

        docker_container_id = ""
        status = "running"

        if self.docker_client:
            network_mode = "none"
            if isinstance(request.network_policy, NetworkPolicyAllowlist):
                network_mode = "bridge"

            try:
                dc = await asyncio.to_thread(
                    self.docker_client.containers.run,
                    image=self.config.default_image,
                    command="sleep infinity",
                    detach=True,
                    mem_limit=memory_limit,
                    network_mode=network_mode,
                    stdin_open=True,
                    tty=True,
                )
                docker_container_id = dc.id
            except Exception as e:
                logger.error("Failed to create Docker container", error=str(e))
                raise RuntimeError(f"Failed to create Docker container: {e}") from e

        network_policy_json = ""
        if request.network_policy:
            network_policy_json = request.network_policy.model_dump_json()

        await self.sql_store.insert(
            "containers",
            {
                "id": container_id,
                "name": request.name,
                "status": status,
                "created_at": now,
                "last_active_at": now,
                "expires_after_minutes": request.expires_after.minutes if request.expires_after else None,
                "memory_limit": memory_limit,
                "network_policy_json": network_policy_json,
                "docker_container_id": docker_container_id,
            },
        )

        return Container(
            id=container_id,
            name=request.name,
            status=status,
            created_at=now,
            last_active_at=now,
            expires_after=request.expires_after,
            memory_limit=memory_limit,
            network_policy=request.network_policy,
        )

    async def list_containers(self, request: ListContainersRequest) -> ListContainersResponse:
        if not self.sql_store:
            raise RuntimeError("Failed to list containers: provider not initialized")

        where = {}
        if request.name:
            where["name"] = request.name

        paginated = await self.sql_store.fetch_all(
            table="containers",
            where=where if where else None,
            order_by=[("created_at", "desc" if request.order is None or request.order.value == "desc" else "asc")],
            cursor=("id", request.after) if request.after else None,
            limit=request.limit or 20,
        )

        containers = [self._row_to_container(row) for row in paginated.data]

        return ListContainersResponse(
            data=containers,
            has_more=paginated.has_more,
            first_id=containers[0].id if containers else "",
            last_id=containers[-1].id if containers else "",
        )

    async def retrieve_container(self, request: RetrieveContainerRequest) -> Container:
        if not self.sql_store:
            raise RuntimeError("Failed to retrieve container: provider not initialized")

        row = await self.sql_store.fetch_one("containers", where={"id": request.container_id})
        if not row:
            raise ResourceNotFoundError(f"Failed to retrieve container: container '{request.container_id}' not found")

        return self._row_to_container(row)

    async def delete_container(self, request: DeleteContainerRequest) -> DeleteContainerResponse:
        if not self.sql_store:
            raise RuntimeError("Failed to delete container: provider not initialized")

        row = await self.sql_store.fetch_one("containers", where={"id": request.container_id})
        if not row:
            raise ResourceNotFoundError(f"Failed to delete container: container '{request.container_id}' not found")

        if self.docker_client and row.get("docker_container_id"):
            try:
                dc = await asyncio.to_thread(self.docker_client.containers.get, row["docker_container_id"])
                await asyncio.to_thread(dc.stop, timeout=10)
                await asyncio.to_thread(dc.remove)
            except Exception as e:
                logger.warning("Failed to remove Docker container", error=str(e))

        await self.sql_store.delete("containers", where={"id": request.container_id})

        return DeleteContainerResponse(id=request.container_id)

    async def exec_in_container(self, request: ExecInContainerRequest) -> ExecInContainerResponse:
        if not self.sql_store:
            raise RuntimeError("Failed to exec in container: provider not initialized")

        row = await self.sql_store.fetch_one("containers", where={"id": request.container_id})
        if not row:
            raise ResourceNotFoundError(f"Failed to exec in container: container '{request.container_id}' not found")

        if not self.docker_client or not row.get("docker_container_id"):
            raise RuntimeError("Failed to exec in container: Docker is not available")

        dc = await asyncio.to_thread(self.docker_client.containers.get, row["docker_container_id"])
        combined_command = " && ".join(request.commands)
        timeout_seconds = (request.timeout_ms or 120000) / 1000.0

        timed_out = False
        try:
            exec_result = await asyncio.wait_for(
                asyncio.to_thread(dc.exec_run, ["sh", "-c", combined_command], demux=True),
                timeout=timeout_seconds,
            )
            exit_code = exec_result.exit_code
            stdout_bytes, stderr_bytes = exec_result.output
            stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
            stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        except TimeoutError:
            timed_out = True
            exit_code = -1
            stdout = ""
            stderr = ""

        if request.max_output_length:
            stdout = stdout[: request.max_output_length]
            stderr = stderr[: request.max_output_length]

        now = int(time.time())
        await self.sql_store.update("containers", {"last_active_at": now}, where={"id": request.container_id})

        return ExecInContainerResponse(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )
