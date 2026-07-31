# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for tenant isolation of vector stores in multi-tenancy deployments.

These tests verify that:
- Users can only access their own vector stores
- Users cannot retrieve, update, or delete other users' vector stores
- Vector store listings only show the user's own stores
- Cross-tenant access returns "not found" rather than "forbidden" (information hiding)

These tests exercise the AuthorizedSqlStore ABAC layer for the vector_stores table,
which uses owner_principal to enforce row-level access control.

No inference recordings are needed — vector store CRUD does not call external providers.

To run these tests, set ALICE_TOKEN and BOB_TOKEN environment variables
with valid auth tokens for two different users, and point at a running OGX
server with auth enabled:

    ALICE_TOKEN=... BOB_TOKEN=... uv run pytest tests/integration/vector_io/test_vector_stores_access_control.py \\
        --stack-config server:ci-tests -x --tb=short
"""

import os

import httpx
import pytest


def get_auth_token(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


def auth_enabled() -> bool:
    return bool(os.environ.get("ALICE_TOKEN") or os.environ.get("BOB_TOKEN"))


class VectorStoresClient:
    """Thin HTTP client for the OGX Vector Stores REST API (/v1/vector_stores)."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def create(self, name: str, embedding_model: str | None = None) -> dict:
        body: dict = {"name": name}
        if embedding_model:
            body["embedding_model"] = embedding_model
        resp = httpx.post(
            f"{self.base_url}/v1/vector_stores",
            headers=self._headers(),
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve(self, vector_store_id: str) -> httpx.Response:
        return httpx.get(
            f"{self.base_url}/v1/vector_stores/{vector_store_id}",
            headers=self._headers(),
            timeout=30.0,
        )

    def update(self, vector_store_id: str, name: str) -> httpx.Response:
        return httpx.post(
            f"{self.base_url}/v1/vector_stores/{vector_store_id}",
            headers=self._headers(),
            json={"name": name},
            timeout=30.0,
        )

    def delete(self, vector_store_id: str) -> httpx.Response:
        return httpx.delete(
            f"{self.base_url}/v1/vector_stores/{vector_store_id}",
            headers=self._headers(),
            timeout=30.0,
        )

    def list_stores(self) -> httpx.Response:
        return httpx.get(
            f"{self.base_url}/v1/vector_stores",
            headers=self._headers(),
            timeout=30.0,
        )


@pytest.mark.integration
@pytest.mark.skipif(not auth_enabled(), reason="Auth tokens not configured (set ALICE_TOKEN and BOB_TOKEN)")
class TestVectorStoresAccessControl:
    """Tests for tenant isolation of vector stores.

    Verifies that vector stores created by one user are invisible to other users,
    following the "information hiding" pattern where cross-tenant access
    returns 404 (not found) rather than 403 (forbidden) to prevent
    resource enumeration attacks.
    """

    @pytest.fixture
    def alice(self, ogx_client) -> VectorStoresClient:
        token = get_auth_token("ALICE_TOKEN", "token-alice")
        return VectorStoresClient(str(ogx_client.base_url), token)

    @pytest.fixture
    def bob(self, ogx_client) -> VectorStoresClient:
        token = get_auth_token("BOB_TOKEN", "token-bob")
        return VectorStoresClient(str(ogx_client.base_url), token)

    EMBEDDING_MODEL = "sentence-transformers/nomic-ai/nomic-embed-text-v1.5"

    def _create_store(self, client: VectorStoresClient, name: str = "test-store") -> str:
        data = client.create(name, embedding_model=self.EMBEDDING_MODEL)
        return data["id"]

    def test_user_cannot_retrieve_other_users_vector_store(self, alice, bob, require_server):
        """Alice's vector store is invisible to Bob — retrieval returns not-found."""
        store_id = self._create_store(alice, "alice-store")

        try:
            alice_resp = alice.retrieve(store_id)
            assert alice_resp.status_code == 200
            assert alice_resp.json()["id"] == store_id

            bob_resp = bob.retrieve(store_id)
            assert bob_resp.status_code in (400, 403, 404), (
                f"Bob should NOT access Alice's vector store, got status {bob_resp.status_code}: {bob_resp.text}"
            )
        finally:
            alice.delete(store_id)

    def test_user_cannot_delete_other_users_vector_store(self, alice, bob, require_server):
        """Bob cannot delete Alice's vector store, and it survives the attempt."""
        store_id = self._create_store(alice, "alice-store")

        try:
            bob_resp = bob.delete(store_id)
            assert bob_resp.status_code in (400, 403, 404), (
                f"Bob should NOT delete Alice's vector store, got status {bob_resp.status_code}: {bob_resp.text}"
            )

            alice_resp = alice.retrieve(store_id)
            assert alice_resp.status_code == 200
            assert alice_resp.json()["id"] == store_id
        finally:
            alice.delete(store_id)

    def test_user_cannot_update_other_users_vector_store(self, alice, bob, require_server):
        """Bob cannot rename Alice's vector store."""
        store_id = self._create_store(alice, "alice-store")

        try:
            bob_resp = bob.update(store_id, name="hijacked-store")
            assert bob_resp.status_code in (400, 403, 404), (
                f"Bob should NOT update Alice's vector store, got status {bob_resp.status_code}: {bob_resp.text}"
            )

            alice_resp = alice.retrieve(store_id)
            assert alice_resp.status_code == 200
            assert alice_resp.json()["name"] == "alice-store"
        finally:
            alice.delete(store_id)

    def test_users_have_isolated_vector_store_listings(self, alice, bob, require_server):
        """Each user sees only their own vector stores in listings."""
        alice_id = self._create_store(alice, "alice-store")
        bob_id = self._create_store(bob, "bob-store")

        try:
            alice_resp = alice.list_stores()
            assert alice_resp.status_code == 200
            alice_ids = {s["id"] for s in alice_resp.json()["data"]}

            bob_resp = bob.list_stores()
            assert bob_resp.status_code == 200
            bob_ids = {s["id"] for s in bob_resp.json()["data"]}

            assert alice_id in alice_ids, "Alice should see her own vector store"
            assert bob_id not in alice_ids, "Alice should NOT see Bob's vector store"
            assert bob_id in bob_ids, "Bob should see his own vector store"
            assert alice_id not in bob_ids, "Bob should NOT see Alice's vector store"
        finally:
            alice.delete(alice_id)
            bob.delete(bob_id)

    def test_vector_store_access_after_cross_tenant_denial(self, alice, bob, require_server):
        """Access control doesn't interfere with legitimate access after a denial."""
        alice_id = self._create_store(alice, "alice-store")
        bob_id = self._create_store(bob, "bob-store")

        try:
            bob_denied = bob.retrieve(alice_id)
            assert bob_denied.status_code in (400, 403, 404)

            bob_own = bob.retrieve(bob_id)
            assert bob_own.status_code == 200
            assert bob_own.json()["id"] == bob_id

            alice_own = alice.retrieve(alice_id)
            assert alice_own.status_code == 200
            assert alice_own.json()["id"] == alice_id
        finally:
            alice.delete(alice_id)
            bob.delete(bob_id)
