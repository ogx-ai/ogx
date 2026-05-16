# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAI Containers API conformance tests.

These tests validate that OGX's Container models match OpenAI's real API
response shapes. Run against OpenAI to record, then replay in CI.

Usage:
    # Record against OpenAI (requires OPENAI_API_KEY)
    OPENAI_API_KEY=sk-... pytest tests/integration/containers/test_openai_conformance.py -v

    # After recording, use the captured response shapes to validate OGX
"""

import os

import pytest

openai_api_key = os.environ.get("OPENAI_API_KEY")
skip_reason = "OPENAI_API_KEY not set"


@pytest.fixture
def real_openai_client():
    if not openai_api_key:
        pytest.skip(skip_reason)
    from openai import OpenAI

    return OpenAI(api_key=openai_api_key)


class TestOpenAIContainersConformance:
    """Record and validate OpenAI Container response shapes."""

    def test_create_and_inspect_container(self, real_openai_client):
        container = real_openai_client.containers.create(
            name="ogx-conformance-test",
            memory_limit="1g",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            obj = container.model_dump()

            assert obj["object"] == "container"
            assert isinstance(obj["id"], str)
            assert obj["id"].startswith("cntr_")
            assert obj["name"] == "ogx-conformance-test"
            assert isinstance(obj["created_at"], int)
            assert obj["status"] in ("running", "active")
            assert "expires_after" in obj
            assert "memory_limit" in obj
        finally:
            real_openai_client.containers.delete(container.id)

    def test_list_containers_shape(self, real_openai_client):
        container = real_openai_client.containers.create(
            name="ogx-conformance-list",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            result = real_openai_client.containers.list()
            obj = result.model_dump()

            assert obj["object"] == "list"
            assert isinstance(obj["data"], list)
            assert "has_more" in obj
            assert "first_id" in obj
            assert "last_id" in obj

            if obj["data"]:
                first = obj["data"][0]
                assert "id" in first
                assert "object" in first
                assert first["object"] == "container"
        finally:
            real_openai_client.containers.delete(container.id)

    def test_retrieve_container_shape(self, real_openai_client):
        container = real_openai_client.containers.create(
            name="ogx-conformance-retrieve",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            retrieved = real_openai_client.containers.retrieve(container.id)
            obj = retrieved.model_dump()

            assert obj["id"] == container.id
            assert obj["object"] == "container"
            assert obj["name"] == "ogx-conformance-retrieve"
            assert isinstance(obj["created_at"], int)
        finally:
            real_openai_client.containers.delete(container.id)

    def test_delete_container(self, real_openai_client):
        container = real_openai_client.containers.create(
            name="ogx-conformance-delete",
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        result = real_openai_client.containers.delete(container.id)
        # OpenAI returns None for successful deletes
        if result is not None:
            obj = result.model_dump()
            assert obj["object"] == "container.deleted"
            assert obj["deleted"] is True
            assert obj["id"] == container.id

    def test_create_with_network_disabled(self, real_openai_client):
        container = real_openai_client.containers.create(
            name="ogx-conformance-network",
            network_policy={"type": "disabled"},
            expires_after={"anchor": "last_active_at", "minutes": 5},
        )
        try:
            obj = container.model_dump()
            assert obj["id"].startswith("cntr_")
        finally:
            real_openai_client.containers.delete(container.id)

    def test_nonexistent_container_raises_not_found(self, real_openai_client):
        from openai import NotFoundError

        with pytest.raises(NotFoundError):
            real_openai_client.containers.retrieve("cntr_nonexistent_000000000000")
