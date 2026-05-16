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
from ogx.providers.inline.containers.docker.config import DockerContainersImplConfig
from ogx.providers.inline.containers.docker.containers import DockerContainersImpl
from ogx_api import ResourceNotFoundError
from ogx_api.containers.models import (
    CreateContainerRequest,
    DeleteContainerRequest,
    ListContainersRequest,
    RetrieveContainerRequest,
)


@pytest.fixture
async def containers_provider(tmp_path):
    db_path = tmp_path / "containers_metadata.db"
    backend_name = f"sql_containers_test_{id(tmp_path)}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})
    config = DockerContainersImplConfig(
        metadata_store=SqlStoreReference(backend=backend_name, table_name="containers_metadata"),
    )
    with patch("ogx.providers.inline.containers.docker.containers.docker") as mock_docker:
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        def make_mock_container(*args, **kwargs):
            mc = MagicMock()
            mc.id = f"docker_{id(mc)}"
            mc.status = "running"
            return mc

        mock_client.containers.run.side_effect = make_mock_container
        mock_client.containers.get.return_value = MagicMock(status="running")

        provider = DockerContainersImpl(config, default_policy())
        await provider.initialize()
        yield provider


class TestDockerContainersProvider:
    async def test_create_container(self, containers_provider):
        req = CreateContainerRequest(name="test-container")
        result = await containers_provider.create_container(req)
        assert result.name == "test-container"
        assert result.object == "container"
        assert result.status == "running"
        assert result.id.startswith("cntr")

    async def test_create_container_with_memory_limit(self, containers_provider):
        req = CreateContainerRequest(name="big-container", memory_limit="4g")
        result = await containers_provider.create_container(req)
        assert result.memory_limit == "4g"

    async def test_list_containers_empty(self, containers_provider):
        req = ListContainersRequest()
        result = await containers_provider.list_containers(req)
        assert result.data == []
        assert result.has_more is False

    async def test_list_containers_after_create(self, containers_provider):
        await containers_provider.create_container(CreateContainerRequest(name="c1"))
        await containers_provider.create_container(CreateContainerRequest(name="c2"))
        req = ListContainersRequest()
        result = await containers_provider.list_containers(req)
        assert len(result.data) == 2

    async def test_retrieve_container(self, containers_provider):
        created = await containers_provider.create_container(CreateContainerRequest(name="findme"))
        req = RetrieveContainerRequest(container_id=created.id)
        result = await containers_provider.retrieve_container(req)
        assert result.id == created.id
        assert result.name == "findme"

    async def test_retrieve_nonexistent_container(self, containers_provider):
        req = RetrieveContainerRequest(container_id="cntr_nonexistent")
        with pytest.raises(ResourceNotFoundError):
            await containers_provider.retrieve_container(req)

    async def test_delete_container(self, containers_provider):
        created = await containers_provider.create_container(CreateContainerRequest(name="deleteme"))
        req = DeleteContainerRequest(container_id=created.id)
        result = await containers_provider.delete_container(req)
        assert result.deleted is True
        assert result.id == created.id

    async def test_delete_nonexistent_container(self, containers_provider):
        req = DeleteContainerRequest(container_id="cntr_nonexistent")
        with pytest.raises(ResourceNotFoundError):
            await containers_provider.delete_container(req)

    async def test_list_with_name_filter(self, containers_provider):
        await containers_provider.create_container(CreateContainerRequest(name="alpha"))
        await containers_provider.create_container(CreateContainerRequest(name="beta"))
        req = ListContainersRequest(name="alpha")
        result = await containers_provider.list_containers(req)
        assert len(result.data) == 1
        assert result.data[0].name == "alpha"
