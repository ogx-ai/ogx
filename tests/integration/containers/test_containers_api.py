# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest


@pytest.fixture(autouse=True)
def skip_if_no_containers_provider(ogx_client):
    providers = [p for p in ogx_client.providers.list() if p.api == "containers"]
    if not providers:
        pytest.skip("No containers provider configured")


class TestContainersAPI:
    def test_create_container(self, openai_client):
        container = openai_client.containers.create(
            name="test-create",
            memory_limit="1g",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            assert container.id.startswith("cntr_") or container.id.startswith("cntr-")
            assert container.object == "container"
            assert container.name == "test-create"
            assert container.status in ("running", "active")
            assert isinstance(container.created_at, int)
        finally:
            openai_client.containers.delete(container.id)

    def test_create_container_minimal(self, openai_client):
        container = openai_client.containers.create(
            name="test-minimal",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            assert container.id is not None
            assert container.name == "test-minimal"
        finally:
            openai_client.containers.delete(container.id)

    def test_list_containers(self, openai_client):
        container = openai_client.containers.create(
            name="test-list",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            result = openai_client.containers.list()
            ids = [c.id for c in result.data]
            assert container.id in ids
            assert result.object == "list"
        finally:
            openai_client.containers.delete(container.id)

    def test_retrieve_container(self, openai_client):
        container = openai_client.containers.create(
            name="test-retrieve",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            retrieved = openai_client.containers.retrieve(container.id)
            assert retrieved.id == container.id
            assert retrieved.name == "test-retrieve"
            assert retrieved.object == "container"
        finally:
            openai_client.containers.delete(container.id)

    def test_delete_container(self, openai_client):
        container = openai_client.containers.create(
            name="test-delete",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        result = openai_client.containers.delete(container.id)
        if result is not None:
            assert result.deleted is True
            assert result.id == container.id
            assert result.object == "container.deleted"

    def test_create_with_memory_limit(self, openai_client):
        container = openai_client.containers.create(
            name="test-memory",
            memory_limit="4g",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            assert container.memory_limit == "4g"
        finally:
            openai_client.containers.delete(container.id)

    def test_create_with_network_policy_disabled(self, openai_client):
        container = openai_client.containers.create(
            name="test-net-disabled",
            network_policy={"type": "disabled"},
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            assert container.id is not None
        finally:
            openai_client.containers.delete(container.id)

    def test_retrieve_nonexistent_raises(self, openai_client):
        from openai import NotFoundError

        with pytest.raises(NotFoundError):
            openai_client.containers.retrieve("cntr_nonexistent_12345")

    def test_list_response_shape(self, openai_client):
        """Validate list response matches OpenAI's pagination shape."""
        container = openai_client.containers.create(
            name="test-shape",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            result = openai_client.containers.list()
            obj = result.model_dump()
            assert obj["object"] == "list"
            assert "data" in obj
            assert "has_more" in obj
            assert "first_id" in obj
            assert "last_id" in obj
        finally:
            openai_client.containers.delete(container.id)

    def test_container_object_fields(self, openai_client):
        """Validate Container object has all required OpenAI fields."""
        container = openai_client.containers.create(
            name="test-fields",
            memory_limit="1g",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            obj = container.model_dump()
            assert "id" in obj
            assert "object" in obj
            assert obj["object"] == "container"
            assert "name" in obj
            assert "created_at" in obj
            assert isinstance(obj["created_at"], int)
            assert "status" in obj
            assert "expires_after" in obj
            assert "memory_limit" in obj
        finally:
            openai_client.containers.delete(container.id)
