# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import pathlib
import time
import uuid

try:
    import openshell as _openshell
except ImportError:
    _openshell = None  # type: ignore[assignment]

try:
    import grpc as _grpc
except ImportError:
    _grpc = None  # type: ignore[assignment]

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

from .config import OpenShellContainersImplConfig

logger = get_logger(name=__name__, category="containers")


_TABLE = "openshell_containers"


class OpenShellContainersImpl(Containers):
    """Containers provider that manages sandboxed environments via OpenShell."""

    def __init__(self, config: OpenShellContainersImplConfig, policy: list[AccessRule]) -> None:
        self.config = config
        self.policy = policy
        self.sql_store: AuthorizedSqlStore | None = None
        self.sandbox_client = None

    async def initialize(self) -> None:
        if _openshell is None:
            raise RuntimeError("Failed to initialize: openshell package is required. Install with: pip install openshell")

        if self.config.gateway_endpoint:
            tls_config = None
            if self.config.tls_ca_path and self.config.tls_cert_path and self.config.tls_key_path:
                tls_config = _openshell.TlsConfig(
                    ca_path=pathlib.Path(self.config.tls_ca_path),
                    cert_path=pathlib.Path(self.config.tls_cert_path),
                    key_path=pathlib.Path(self.config.tls_key_path),
                )
            self.sandbox_client = _openshell.SandboxClient(
                self.config.gateway_endpoint,
                tls=tls_config,
                timeout=self.config.grpc_timeout,
            )
        else:
            self.sandbox_client = await asyncio.to_thread(
                _openshell.SandboxClient.from_active_cluster,
                cluster=self.config.cluster_name,
                timeout=self.config.grpc_timeout,
            )

        try:
            await asyncio.to_thread(self.sandbox_client.health)
        except Exception as e:
            logger.warning("Failed to connect to OpenShell gateway", error=str(e))

        self.sql_store = await authorized_sqlstore(self.config.metadata_store, self.policy)
        await self.sql_store.create_table(
            _TABLE,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "name": ColumnType.STRING,
                "status": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "last_active_at": ColumnType.INTEGER,
                "expires_after_minutes": ColumnType.INTEGER,
                "memory_limit": ColumnType.STRING,
                "network_policy_json": ColumnType.STRING,
                "openshell_sandbox_name": ColumnType.STRING,
            },
        )

    async def shutdown(self) -> None:
        if self.sandbox_client:
            await asyncio.to_thread(self.sandbox_client.close)

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
        if not self.sql_store or not self.sandbox_client:
            raise RuntimeError("Failed to create container: provider not initialized")

        container_id = generate_object_id("cntr", lambda: f"cntr-{uuid.uuid4().hex}")
        now = int(time.time())
        memory_limit = request.memory_limit or "1g"

        try:
            sandbox_ref = await asyncio.to_thread(self.sandbox_client.create)
            await asyncio.to_thread(
                self.sandbox_client.wait_ready,
                sandbox_ref.name,
                timeout_seconds=self.config.ready_timeout_seconds,
            )
        except Exception as e:
            logger.error("Failed to create OpenShell sandbox", error=str(e))
            raise RuntimeError(f"Failed to create OpenShell sandbox: {e}") from e

        network_policy_json = ""
        if request.network_policy:
            network_policy_json = request.network_policy.model_dump_json()

        await self.sql_store.insert(
            _TABLE,
            {
                "id": container_id,
                "name": request.name,
                "status": "running",
                "created_at": now,
                "last_active_at": now,
                "expires_after_minutes": request.expires_after.minutes if request.expires_after else None,
                "memory_limit": memory_limit,
                "network_policy_json": network_policy_json,
                "openshell_sandbox_name": sandbox_ref.name,
            },
        )

        return Container(
            id=container_id,
            name=request.name,
            status="running",
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
            table=_TABLE,
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

        row = await self.sql_store.fetch_one(_TABLE, where={"id": request.container_id})
        if not row:
            raise ResourceNotFoundError(f"Failed to retrieve container: container '{request.container_id}' not found")

        return self._row_to_container(row)

    async def delete_container(self, request: DeleteContainerRequest) -> DeleteContainerResponse:
        if not self.sql_store:
            raise RuntimeError("Failed to delete container: provider not initialized")

        row = await self.sql_store.fetch_one(_TABLE, where={"id": request.container_id})
        if not row:
            raise ResourceNotFoundError(f"Failed to delete container: container '{request.container_id}' not found")

        if self.sandbox_client and row.get("openshell_sandbox_name"):
            try:
                await asyncio.to_thread(self.sandbox_client.delete, row["openshell_sandbox_name"])
            except Exception as e:
                logger.warning("Failed to delete OpenShell sandbox", error=str(e))

        await self.sql_store.delete(_TABLE, where={"id": request.container_id})

        return DeleteContainerResponse(id=request.container_id)

    async def exec_in_container(self, request: ExecInContainerRequest) -> ExecInContainerResponse:
        if not self.sql_store or not self.sandbox_client:
            raise RuntimeError("Failed to exec in container: provider not initialized")

        row = await self.sql_store.fetch_one(_TABLE, where={"id": request.container_id})
        if not row:
            raise ResourceNotFoundError(f"Failed to exec in container: container '{request.container_id}' not found")

        sandbox_name = row.get("openshell_sandbox_name")
        if not sandbox_name:
            raise RuntimeError("Failed to exec in container: no OpenShell sandbox associated")

        sandbox_ref = await asyncio.to_thread(self.sandbox_client.get, sandbox_name)
        combined_command = " && ".join(request.commands)
        timeout_seconds = int((request.timeout_ms or 120000) / 1000)

        timed_out = False
        try:
            result = await asyncio.to_thread(
                self.sandbox_client.exec,
                sandbox_ref.id,
                ["sh", "-c", combined_command],
                timeout_seconds=timeout_seconds,
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.exit_code
        except Exception as e:
            if _grpc is not None and isinstance(e, _grpc.RpcError) and hasattr(e, "code"):
                if e.code() == _grpc.StatusCode.DEADLINE_EXCEEDED:
                    timed_out = True
                    stdout = ""
                    stderr = ""
                    exit_code = -1
                else:
                    raise RuntimeError(f"Failed to exec in container: {e}") from e
            else:
                raise RuntimeError(f"Failed to exec in container: {e}") from e

        if request.max_output_length:
            stdout = stdout[: request.max_output_length]
            stderr = stderr[: request.max_output_length]

        now = int(time.time())
        await self.sql_store.update(_TABLE, {"last_active_at": now}, where={"id": request.container_id})

        return ExecInContainerResponse(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )
