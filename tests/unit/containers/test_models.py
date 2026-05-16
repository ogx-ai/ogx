# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx_api.containers.models import (
    Container,
    ContainerExpiresAfter,
    CreateContainerRequest,
    DeleteContainerRequest,
    DeleteContainerResponse,
    ListContainersRequest,
    ListContainersResponse,
    NetworkPolicyAllowlist,
    NetworkPolicyDisabled,
    RetrieveContainerRequest,
)
from ogx_api.common.responses import Order


class TestContainerModels:
    def test_create_container_request_minimal(self):
        req = CreateContainerRequest(name="test")
        assert req.name == "test"
        assert req.memory_limit is None
        assert req.expires_after is None
        assert req.network_policy is None

    def test_create_container_request_full(self):
        req = CreateContainerRequest(
            name="analysis",
            memory_limit="4g",
            expires_after=ContainerExpiresAfter(anchor="last_active_at", minutes=20),
            network_policy=NetworkPolicyAllowlist(
                allowed_domains=["pypi.org", "files.pythonhosted.org"],
            ),
        )
        assert req.memory_limit == "4g"
        assert req.expires_after.minutes == 20
        assert req.network_policy.type == "allowlist"
        assert len(req.network_policy.allowed_domains) == 2

    def test_create_container_request_invalid_memory_limit(self):
        with pytest.raises(Exception):
            CreateContainerRequest(name="test", memory_limit="99g")

    def test_container_object(self):
        c = Container(
            id="cntr_abc123",
            name="test",
            created_at=1700000000,
            status="running",
        )
        assert c.object == "container"
        assert c.status == "running"

    def test_list_containers_request_defaults(self):
        req = ListContainersRequest()
        assert req.limit == 20
        assert req.order == Order.desc
        assert req.after is None
        assert req.name is None

    def test_delete_container_response(self):
        resp = DeleteContainerResponse(id="cntr_abc123")
        assert resp.object == "container.deleted"
        assert resp.deleted is True

    def test_network_policy_disabled(self):
        policy = NetworkPolicyDisabled()
        assert policy.type == "disabled"

    def test_network_policy_allowlist(self):
        policy = NetworkPolicyAllowlist(allowed_domains=["example.com"])
        assert policy.type == "allowlist"
        assert policy.allowed_domains == ["example.com"]

    def test_expires_after(self):
        ea = ContainerExpiresAfter(anchor="last_active_at", minutes=30)
        assert ea.anchor == "last_active_at"
        assert ea.minutes == 30
