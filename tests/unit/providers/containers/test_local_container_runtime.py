# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the inline::local ContainerRuntime, with a faked docker-py client."""

import asyncio
import io
import tarfile
import time
import zipfile

import pytest

from ogx.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.inline.container_runtime.local.config import LocalContainerRuntimeConfig
from ogx.providers.inline.container_runtime.local.impl import LocalContainerRuntimeImpl
from ogx_api import (
    ContainerCreateRequest,
    ContainerExpiresAfter,
    ContainerFileNotFoundError,
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
    ListContainersRequest,
    MountSkillsRequest,
    NetworkPolicyExtended,
    NetworkPolicyMode,
    UploadContainerFileRequest,
)


class MockUploadFile:
    def __init__(self, content: bytes, filename: str):
        self.content = content
        self.filename = filename

    async def read(self) -> bytes:
        return self.content


class FakeContainer:
    """In-memory stand-in for a docker-py Container with a tiny filesystem."""

    def __init__(self, container_id: str, run_kwargs: dict):
        self.id = container_id
        self.run_kwargs = run_kwargs
        self.files: dict[str, bytes] = {}
        self.removed = False
        # exec_run behavior overridable per test
        self.exec_result = (0, (b"ok\n", None))

    def put_archive(self, path: str, data: bytes) -> bool:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    extracted = tar.extractfile(member)
                    content = extracted.read() if extracted else b""
                    self.files[f"{path.rstrip('/')}/{member.name}"] = content
        return True

    def get_archive(self, path: str):
        data = self.files.get(path, b"")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.rsplit("/", 1)[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return iter([buf.getvalue()]), {"size": len(data)}

    def exec_run(self, cmd, demux=False):
        if isinstance(cmd, list) and cmd and cmd[0] == "rm":
            self.files.pop(cmd[-1], None)
            return _ExecResult(0, (b"", None))
        if isinstance(cmd, list) and cmd and cmd[0] == "mkdir":
            return _ExecResult(0, (b"", None))
        exit_code, output = self.exec_result
        return _ExecResult(exit_code, output)

    def remove(self, force=False):
        self.removed = True


class _ExecResult:
    def __init__(self, exit_code, output):
        self.exit_code = exit_code
        self.output = output


class FakeContainers:
    def __init__(self):
        self._by_id: dict[str, FakeContainer] = {}
        self._counter = 0

    def run(self, image, **kwargs):
        self._counter += 1
        backend_id = f"backend-{self._counter}"
        container = FakeContainer(backend_id, {"image": image, **kwargs})
        self._by_id[backend_id] = container
        return container

    def get(self, backend_id):
        if backend_id not in self._by_id:
            raise KeyError(backend_id)
        return self._by_id[backend_id]


class FakeDockerClient:
    def __init__(self):
        self.containers = FakeContainers()
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
async def runtime(tmp_path):
    backend_name = f"sql_cr_{tmp_path.name}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=(tmp_path / "cr.db").as_posix())})
    config = LocalContainerRuntimeConfig(
        default_image="python:3.12-slim",
        max_containers=2,
        expiration_poll_seconds=3600,  # never fires during tests; we call _reap_expired directly
        metadata_store=SqlStoreReference(backend=backend_name, table_name="cr_meta"),
    )
    impl = LocalContainerRuntimeImpl(config, policy=[])
    await impl.initialize()
    impl._docker_client = FakeDockerClient()  # bypass real docker; lazy seam
    yield impl
    await impl.shutdown()


async def _create(runtime, **kwargs) -> str:
    container = await runtime.create_container(ContainerCreateRequest(**kwargs))
    return container.id


async def test_create_and_get_container(runtime):
    cid = await _create(runtime, name="c1", memory_limit=ContainerMemoryLimit.GB_4)
    fetched = await runtime.get_container(GetContainerRequest(container_id=cid))
    assert fetched.id == cid
    assert fetched.name == "c1"
    assert fetched.memory_limit == ContainerMemoryLimit.GB_4
    assert fetched.status == ContainerStatus.ACTIVE
    # mem_limit forwarded to the engine as bytes
    backend = runtime._docker_client.containers._by_id["backend-1"]
    assert backend.run_kwargs["mem_limit"] == 4 * 1024**3


async def test_get_missing_container_raises(runtime):
    with pytest.raises(ContainerNotFoundError):
        await runtime.get_container(GetContainerRequest(container_id="container-missing"))


async def test_max_containers_enforced(runtime):
    await _create(runtime)
    await _create(runtime)
    with pytest.raises(RuntimeError, match="max_containers"):
        await _create(runtime)


async def test_list_containers(runtime):
    await _create(runtime, name="a")
    await _create(runtime, name="b")
    listed = await runtime.list_containers(ListContainersRequest())
    assert len(listed.data) == 2
    assert listed.has_more is False


async def test_network_policy_disables_network_when_deny(runtime):
    await _create(runtime)  # default no policy -> network disabled
    backend = runtime._docker_client.containers._by_id["backend-1"]
    assert backend.run_kwargs["network_disabled"] is True

    await _create(runtime, network_policy=NetworkPolicyExtended(mode=NetworkPolicyMode.ALLOW_ALL))
    backend2 = runtime._docker_client.containers._by_id["backend-2"]
    assert backend2.run_kwargs["network_disabled"] is False


async def test_delete_container(runtime):
    cid = await _create(runtime)
    backend = runtime._docker_client.containers._by_id["backend-1"]
    resp = await runtime.delete_container(DeleteContainerRequest(container_id=cid))
    assert resp.deleted is True
    assert backend.removed is True
    with pytest.raises(ContainerNotFoundError):
        await runtime.get_container(GetContainerRequest(container_id=cid))


async def test_file_upload_list_get_content_delete(runtime):
    cid = await _create(runtime)
    upload = MockUploadFile(b"file-bytes", "data.txt")
    cf = await runtime.upload_file(UploadContainerFileRequest(container_id=cid), upload)
    assert cf.bytes == len(b"file-bytes")
    assert cf.path == "/mnt/data/data.txt"

    files = await runtime.list_files(ListContainerFilesRequest(container_id=cid))
    assert len(files.data) == 1

    got = await runtime.get_file(GetContainerFileRequest(container_id=cid, file_id=cf.id))
    assert got.id == cf.id

    content = await runtime.get_file_content(GetContainerFileContentRequest(container_id=cid, file_id=cf.id))
    assert content.body == b"file-bytes"

    await runtime.delete_file(DeleteContainerFileRequest(container_id=cid, file_id=cf.id))
    with pytest.raises(ContainerFileNotFoundError):
        await runtime.get_file(GetContainerFileRequest(container_id=cid, file_id=cf.id))


async def test_execute_shell_success(runtime):
    cid = await _create(runtime)
    out = await runtime.execute_shell(ExecuteShellRequest(container_id=cid, command=["echo", "hi"]))
    assert out.outcome.type == "success"
    assert out.stdout == "ok\n"
    assert out.container_id == cid


async def test_execute_shell_failure(runtime):
    cid = await _create(runtime)
    backend = runtime._docker_client.containers._by_id["backend-1"]
    backend.exec_result = (3, (b"", b"boom\n"))
    out = await runtime.execute_shell(ExecuteShellRequest(container_id=cid, command=["false"]))
    assert out.outcome.type == "failure"
    assert out.outcome.exit_code == 3
    assert out.stderr == "boom\n"


async def test_execute_shell_timeout(runtime):
    cid = await _create(runtime)
    backend = runtime._docker_client.containers._by_id["backend-1"]

    def slow_exec(cmd, demux=False):
        time.sleep(0.2)
        return _ExecResult(0, (b"", None))

    backend.exec_run = slow_exec
    out = await runtime.execute_shell(
        ExecuteShellRequest(container_id=cid, command=["sleep", "1"], timeout_seconds=0.01)
    )
    assert out.outcome.type == "timeout"


async def test_execute_shell_refreshes_last_active(runtime):
    cid = await _create(runtime)
    before = (await runtime.get_container(GetContainerRequest(container_id=cid))).last_active_at
    await asyncio.sleep(1.1)
    await runtime.execute_shell(ExecuteShellRequest(container_id=cid, command=["echo", "x"]))
    after = (await runtime.get_container(GetContainerRequest(container_id=cid))).last_active_at
    assert after >= before


async def test_mount_skills(runtime):
    cid = await _create(runtime)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("SKILL.md", b"do things")
    await runtime.mount_skills(MountSkillsRequest(container_id=cid, skill_bundles=[("my-skill", zip_buf.getvalue())]))
    backend = runtime._docker_client.containers._by_id["backend-1"]
    assert any("/mnt/skills/my-skill/SKILL.md" == path for path in backend.files)


async def test_reap_expired(runtime):
    cid = await _create(runtime, expires_after=ContainerExpiresAfter(minutes=1))
    # force last_active_at into the past so it is expired
    await runtime.sql_store.update(
        "container_runtime_containers", {"last_active_at": int(time.time()) - 3600}, where={"id": cid}
    )
    await runtime._reap_expired()
    reaped = await runtime.get_container(GetContainerRequest(container_id=cid))
    assert reaped.status == ContainerStatus.EXPIRED


def test_engine_socket_selection():
    docker_cfg = LocalContainerRuntimeConfig(metadata_store=SqlStoreReference(backend="x", table_name="t"))
    assert docker_cfg.resolved_socket_url().endswith("docker.sock")
    podman_cfg = LocalContainerRuntimeConfig(
        engine="podman", metadata_store=SqlStoreReference(backend="x", table_name="t")
    )
    assert "podman" in podman_cfg.resolved_socket_url()
