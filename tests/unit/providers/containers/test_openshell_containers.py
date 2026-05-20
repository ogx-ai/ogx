# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest

from ogx.core.access_control.access_control import default_policy
from ogx.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.remote.containers.openshell.config import OpenShellContainersImplConfig
from ogx.providers.remote.containers.openshell.containers import OpenShellContainersImpl
from ogx_api import ResourceNotFoundError
from ogx_api.containers.models import (
    CreateContainerRequest,
    DeleteContainerRequest,
    ExecInContainerRequest,
    ListContainersRequest,
    RetrieveContainerRequest,
)


def _make_mock_sandbox_ref(name="sb-test-001", sandbox_id="sb-id-001"):
    ref = MagicMock()
    ref.name = name
    ref.id = sandbox_id
    ref.phase = 2
    return ref


def _make_mock_exec_result(stdout="hello\n", stderr="", exit_code=0):
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.exit_code = exit_code
    return result


@pytest.fixture
async def openshell_provider(tmp_path):
    db_path = tmp_path / "openshell_containers_metadata.db"
    backend_name = f"sql_openshell_test_{id(tmp_path)}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})
    config = OpenShellContainersImplConfig(
        gateway_endpoint="localhost:50051",
        metadata_store=SqlStoreReference(backend=backend_name, table_name="openshell_containers_metadata"),
    )

    mock_sandbox_ref = _make_mock_sandbox_ref()

    with (
        patch("ogx.providers.remote.containers.openshell.containers._openshell") as mock_openshell,
        patch("ogx.providers.remote.containers.openshell.containers._grpc") as mock_grpc,
    ):
        mock_client = MagicMock()
        mock_openshell.SandboxClient.return_value = mock_client
        mock_openshell.TlsConfig = MagicMock()
        mock_client.health.return_value = MagicMock()
        mock_client.create.return_value = mock_sandbox_ref
        mock_client.wait_ready.return_value = mock_sandbox_ref
        mock_client.get.return_value = mock_sandbox_ref
        mock_client.delete.return_value = True
        mock_client.exec.return_value = _make_mock_exec_result()
        mock_client.close.return_value = None

        provider = OpenShellContainersImpl(config, default_policy())
        await provider.initialize()
        yield provider


class TestOpenShellContainersProvider:
    async def test_create_container(self, openshell_provider):
        req = CreateContainerRequest(name="test-container")
        result = await openshell_provider.create_container(req)
        assert result.name == "test-container"
        assert result.object == "container"
        assert result.status == "running"
        assert result.id.startswith("cntr")

    async def test_create_container_with_memory_limit(self, openshell_provider):
        req = CreateContainerRequest(name="big-container", memory_limit="4g")
        result = await openshell_provider.create_container(req)
        assert result.memory_limit == "4g"

    async def test_list_containers_empty(self, openshell_provider):
        req = ListContainersRequest()
        result = await openshell_provider.list_containers(req)
        assert result.data == []
        assert result.has_more is False

    async def test_list_containers_after_create(self, openshell_provider):
        await openshell_provider.create_container(CreateContainerRequest(name="c1"))
        await openshell_provider.create_container(CreateContainerRequest(name="c2"))
        req = ListContainersRequest()
        result = await openshell_provider.list_containers(req)
        assert len(result.data) == 2

    async def test_retrieve_container(self, openshell_provider):
        created = await openshell_provider.create_container(CreateContainerRequest(name="findme"))
        req = RetrieveContainerRequest(container_id=created.id)
        result = await openshell_provider.retrieve_container(req)
        assert result.id == created.id
        assert result.name == "findme"

    async def test_retrieve_nonexistent_container(self, openshell_provider):
        req = RetrieveContainerRequest(container_id="cntr_nonexistent")
        with pytest.raises(ResourceNotFoundError):
            await openshell_provider.retrieve_container(req)

    async def test_delete_container(self, openshell_provider):
        created = await openshell_provider.create_container(CreateContainerRequest(name="deleteme"))
        req = DeleteContainerRequest(container_id=created.id)
        result = await openshell_provider.delete_container(req)
        assert result.deleted is True
        assert result.id == created.id

    async def test_delete_nonexistent_container(self, openshell_provider):
        req = DeleteContainerRequest(container_id="cntr_nonexistent")
        with pytest.raises(ResourceNotFoundError):
            await openshell_provider.delete_container(req)

    async def test_list_with_name_filter(self, openshell_provider):
        await openshell_provider.create_container(CreateContainerRequest(name="alpha"))
        await openshell_provider.create_container(CreateContainerRequest(name="beta"))
        req = ListContainersRequest(name="alpha")
        result = await openshell_provider.list_containers(req)
        assert len(result.data) == 1
        assert result.data[0].name == "alpha"

    async def test_exec_in_container(self, openshell_provider):
        created = await openshell_provider.create_container(CreateContainerRequest(name="exec-test"))
        req = ExecInContainerRequest(
            container_id=created.id,
            commands=["echo hello", "echo world"],
        )
        result = await openshell_provider.exec_in_container(req)
        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert result.timed_out is False

    async def test_exec_output_truncation(self, openshell_provider):
        openshell_provider.sandbox_client.exec.return_value = _make_mock_exec_result(
            stdout="a" * 1000, stderr="b" * 1000
        )
        created = await openshell_provider.create_container(CreateContainerRequest(name="truncate-test"))
        req = ExecInContainerRequest(
            container_id=created.id,
            commands=["echo long"],
            max_output_length=100,
        )
        result = await openshell_provider.exec_in_container(req)
        assert len(result.stdout) == 100
        assert len(result.stderr) == 100

    async def test_exec_nonexistent_container(self, openshell_provider):
        req = ExecInContainerRequest(container_id="cntr_nonexistent", commands=["echo hi"])
        with pytest.raises(ResourceNotFoundError):
            await openshell_provider.exec_in_container(req)


class TestOpenShellConfig:
    def test_config_defaults(self):
        config = OpenShellContainersImplConfig(
            metadata_store=SqlStoreReference(backend="sql_default", table_name="test"),
        )
        assert config.gateway_endpoint is None
        assert config.cluster_name is None
        assert config.ready_timeout_seconds == 120.0
        assert config.grpc_timeout == 30.0

    def test_config_explicit_endpoint(self):
        config = OpenShellContainersImplConfig(
            gateway_endpoint="openshell.example.com:443",
            tls_ca_path="/etc/ssl/ca.crt",
            tls_cert_path="/etc/ssl/tls.crt",
            tls_key_path="/etc/ssl/tls.key",
            metadata_store=SqlStoreReference(backend="sql_default", table_name="test"),
        )
        assert config.gateway_endpoint == "openshell.example.com:443"
        assert config.tls_ca_path == "/etc/ssl/ca.crt"

    def test_sample_run_config(self):
        sample = OpenShellContainersImplConfig.sample_run_config(__distro_dir__=".")
        assert sample["gateway_endpoint"] == "localhost:50051"
        assert "metadata_store" in sample
