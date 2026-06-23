# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the inline::builtin Containers provider (policy + delegation)."""

from unittest.mock import AsyncMock

import pytest
from fastapi import Response

from ogx.providers.inline.containers.builtin.config import BuiltinContainersConfig
from ogx.providers.inline.containers.builtin.impl import BuiltinContainersImpl
from ogx_api import (
    Container,
    ContainerCreateRequest,
    ContainerFile,
    ContainerStatus,
    InvalidParameterError,
    NetworkPolicy,
    NetworkPolicyExtended,
    NetworkPolicyMode,
    NetworkPolicyViolationError,
    OpenAIFileObject,
)


def _container(cid="container-1") -> Container:
    return Container(id=cid, created_at=1, status=ContainerStatus.ACTIVE, last_active_at=1)


def _make_impl(config: BuiltinContainersConfig | None = None):
    runtime = AsyncMock()
    runtime.create_container.return_value = _container()
    runtime.upload_file.return_value = ContainerFile(
        id="container_file-1", container_id="container-1", created_at=1, bytes=3, path="/mnt/data/x", source="user"
    )
    files = AsyncMock()
    impl = BuiltinContainersImpl(config or BuiltinContainersConfig(), runtime, files, policy=[])
    return impl, runtime, files


async def test_create_delegates_and_passes_effective_policy():
    config = BuiltinContainersConfig(
        default_network_policy=NetworkPolicy(mode=NetworkPolicyMode.ALLOW_LIST, allow_domains=["a.com", "b.com"])
    )
    impl, runtime, _files = _make_impl(config)
    await impl.create_container(
        ContainerCreateRequest(
            network_policy=NetworkPolicyExtended(mode=NetworkPolicyMode.ALLOW_LIST, allow_domains=["a.com"])
        )
    )
    sent = runtime.create_container.call_args.args[0]
    assert sent.network_policy.allow_domains == ["a.com"]
    assert sent.network_policy.mode == NetworkPolicyMode.ALLOW_LIST


async def test_create_rejects_policy_widening():
    config = BuiltinContainersConfig(default_network_policy=NetworkPolicy(mode=NetworkPolicyMode.DENY))
    impl, runtime, _files = _make_impl(config)
    with pytest.raises(NetworkPolicyViolationError):
        await impl.create_container(
            ContainerCreateRequest(network_policy=NetworkPolicyExtended(mode=NetworkPolicyMode.ALLOW_ALL))
        )
    runtime.create_container.assert_not_called()


async def test_create_no_policy_uses_operator_default():
    config = BuiltinContainersConfig(
        default_network_policy=NetworkPolicy(mode=NetworkPolicyMode.ALLOW_LIST, allow_domains=["x.com"])
    )
    impl, runtime, _files = _make_impl(config)
    await impl.create_container(ContainerCreateRequest())
    sent = runtime.create_container.call_args.args[0]
    assert sent.network_policy.mode == NetworkPolicyMode.ALLOW_LIST
    assert sent.network_policy.allow_domains == ["x.com"]


async def test_image_lock_rejects_disallowed_image():
    config = BuiltinContainersConfig(allowed_images=["python:3.12-slim"])
    impl, runtime, _files = _make_impl(config)
    with pytest.raises(InvalidParameterError):
        await impl.create_container(ContainerCreateRequest(image="evil:latest"))
    runtime.create_container.assert_not_called()


async def test_image_lock_allows_listed_image():
    config = BuiltinContainersConfig(allowed_images=["python:3.12-slim"])
    impl, runtime, _files = _make_impl(config)
    await impl.create_container(ContainerCreateRequest(image="python:3.12-slim"))
    runtime.create_container.assert_called_once()


async def test_file_ids_seeded_from_files_api():
    impl, runtime, files = _make_impl()
    files.openai_retrieve_file.return_value = OpenAIFileObject(
        id="file-1", filename="seed.txt", purpose="assistants", bytes=3, created_at=1, expires_at=0, status="processed"
    )
    files.openai_retrieve_file_content.return_value = Response(content=b"abc")

    await impl.create_container(ContainerCreateRequest(file_ids=["file-1"]))

    files.openai_retrieve_file_content.assert_awaited_once()
    runtime.upload_file.assert_awaited_once()
    upload = runtime.upload_file.call_args.args[1]
    assert upload.filename == "seed.txt"


async def test_delegation_methods_forward_to_runtime():
    impl, runtime, _files = _make_impl()
    from ogx_api import GetContainerRequest

    await impl.get_container(GetContainerRequest(container_id="container-1"))
    runtime.get_container.assert_awaited_once()
