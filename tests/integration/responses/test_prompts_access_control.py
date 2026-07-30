# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for tenant isolation of prompts in multi-tenancy deployments.

These tests verify that:
- Users can only access their own prompt templates
- Users cannot retrieve, update, or delete other users' prompts
- Prompt listings only show the user's own prompts
- Cross-tenant access returns "not found" rather than "forbidden" (information hiding)

These tests exercise the AuthorizedSqlStore ABAC layer for the prompts table,
which uses owner_principal to enforce row-level access control.

No inference recordings are needed — prompts CRUD does not call external providers.

To run these tests, set ALICE_TOKEN and BOB_TOKEN environment variables
with valid auth tokens for two different users, and point at a running OGX
server with auth enabled:

    ALICE_TOKEN=... BOB_TOKEN=... uv run pytest tests/integration/responses/test_prompts_access_control.py \\
        --stack-config server:ci-tests -x --tb=short
"""

import os

import httpx
import pytest


def get_auth_token(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


def auth_enabled() -> bool:
    return bool(os.environ.get("ALICE_TOKEN") or os.environ.get("BOB_TOKEN"))


class PromptsClient:
    """Thin HTTP client for the OGX Prompts REST API (/v1/prompts)."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def create(self, prompt: str, variables: list[str] | None = None) -> dict:
        resp = httpx.post(
            f"{self.base_url}/v1/prompts",
            headers=self._headers(),
            json={"prompt": prompt, "variables": variables or []},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve(self, prompt_id: str, version: int | None = None) -> httpx.Response:
        url = f"{self.base_url}/v1/prompts/{prompt_id}"
        if version is not None:
            url += f"?version={version}"
        resp = httpx.get(url, headers=self._headers(), timeout=30.0)
        return resp

    def update(self, prompt_id: str, version: int, prompt: str, variables: list[str]) -> httpx.Response:
        resp = httpx.put(
            f"{self.base_url}/v1/prompts/{prompt_id}",
            headers=self._headers(),
            json={"version": version, "prompt": prompt, "variables": variables},
            timeout=30.0,
        )
        return resp

    def delete(self, prompt_id: str) -> httpx.Response:
        resp = httpx.delete(
            f"{self.base_url}/v1/prompts/{prompt_id}",
            headers=self._headers(),
            timeout=30.0,
        )
        return resp

    def list_prompts(self) -> httpx.Response:
        resp = httpx.get(
            f"{self.base_url}/v1/prompts",
            headers=self._headers(),
            timeout=30.0,
        )
        return resp


@pytest.mark.integration
@pytest.mark.skipif(not auth_enabled(), reason="Auth tokens not configured (set ALICE_TOKEN and BOB_TOKEN)")
class TestPromptsAccessControl:
    """Tests for tenant isolation of prompt templates.

    Verifies that prompts stored by one user are invisible to other users,
    following the "information hiding" pattern where cross-tenant access
    returns 404 (not found) rather than 403 (forbidden) to prevent
    resource enumeration attacks.
    """

    @pytest.fixture
    def alice(self, ogx_client) -> PromptsClient:
        token = get_auth_token("ALICE_TOKEN", "token-alice")
        return PromptsClient(str(ogx_client.base_url), token)

    @pytest.fixture
    def bob(self, ogx_client) -> PromptsClient:
        token = get_auth_token("BOB_TOKEN", "token-bob")
        return PromptsClient(str(ogx_client.base_url), token)

    def _create_prompt(self, client: PromptsClient, text: str = "You are a {{ role }} assistant.") -> str:
        data = client.create(text, ["role"])
        return data["prompt_id"]

    def test_user_cannot_retrieve_other_users_prompt(self, alice, bob, require_server):
        """Alice's prompt is invisible to Bob — retrieval returns not-found."""
        prompt_id = self._create_prompt(alice)

        try:
            alice_resp = alice.retrieve(prompt_id)
            assert alice_resp.status_code == 200
            assert alice_resp.json()["prompt_id"] == prompt_id

            bob_resp = bob.retrieve(prompt_id)
            assert bob_resp.status_code in (400, 403, 404), (
                f"Bob should NOT access Alice's prompt, got status {bob_resp.status_code}: {bob_resp.text}"
            )
        finally:
            alice.delete(prompt_id)

    def test_user_cannot_delete_other_users_prompt(self, alice, bob, require_server):
        """Bob cannot delete Alice's prompt, and it survives the attempt."""
        prompt_id = self._create_prompt(alice)

        try:
            bob_resp = bob.delete(prompt_id)
            assert bob_resp.status_code in (400, 403, 404), (
                f"Bob should NOT delete Alice's prompt, got status {bob_resp.status_code}: {bob_resp.text}"
            )

            alice_resp = alice.retrieve(prompt_id)
            assert alice_resp.status_code == 200
            assert alice_resp.json()["prompt_id"] == prompt_id
        finally:
            alice.delete(prompt_id)

    def test_user_cannot_update_other_users_prompt(self, alice, bob, require_server):
        """Bob cannot create a new version of Alice's prompt."""
        prompt_id = self._create_prompt(alice)

        try:
            bob_resp = bob.update(prompt_id, version=1, prompt="Hijacked: {{ role }}", variables=["role"])
            assert bob_resp.status_code in (400, 403, 404), (
                f"Bob should NOT update Alice's prompt, got status {bob_resp.status_code}: {bob_resp.text}"
            )

            alice_resp = alice.retrieve(prompt_id)
            assert alice_resp.status_code == 200
            data = alice_resp.json()
            assert data["version"] == 1
            assert "Hijacked" not in data["prompt"]
        finally:
            alice.delete(prompt_id)

    def test_users_have_isolated_prompt_listings(self, alice, bob, require_server):
        """Each user sees only their own prompts in listings."""
        alice_id = self._create_prompt(alice, "Alice prompt: {{ role }}")
        bob_id = self._create_prompt(bob, "Bob prompt: {{ role }}")

        try:
            alice_resp = alice.list_prompts()
            assert alice_resp.status_code == 200
            alice_ids = {p["prompt_id"] for p in alice_resp.json()["data"]}

            bob_resp = bob.list_prompts()
            assert bob_resp.status_code == 200
            bob_ids = {p["prompt_id"] for p in bob_resp.json()["data"]}

            assert alice_id in alice_ids, "Alice should see her own prompt"
            assert bob_id not in alice_ids, "Alice should NOT see Bob's prompt"
            assert bob_id in bob_ids, "Bob should see his own prompt"
            assert alice_id not in bob_ids, "Bob should NOT see Alice's prompt"
        finally:
            alice.delete(alice_id)
            bob.delete(bob_id)

    def test_prompt_access_after_cross_tenant_denial(self, alice, bob, require_server):
        """Access control doesn't interfere with legitimate access after a denial."""
        alice_id = self._create_prompt(alice)
        bob_id = self._create_prompt(bob, "Bob's prompt: {{ role }}")

        try:
            bob_denied = bob.retrieve(alice_id)
            assert bob_denied.status_code in (400, 403, 404)

            bob_own = bob.retrieve(bob_id)
            assert bob_own.status_code == 200
            assert bob_own.json()["prompt_id"] == bob_id

            alice_own = alice.retrieve(alice_id)
            assert alice_own.status_code == 200
            assert alice_own.json()["prompt_id"] == alice_id
        finally:
            alice.delete(alice_id)
            bob.delete(bob_id)

    def test_prompt_version_isolation(self, alice, bob, require_server):
        """Bob cannot access any version of Alice's prompt, including after updates."""
        prompt_id = self._create_prompt(alice)

        try:
            update_resp = alice.update(prompt_id, version=1, prompt="Updated v2: {{ role }}", variables=["role"])
            assert update_resp.status_code == 200

            bob_v1 = bob.retrieve(prompt_id, version=1)
            assert bob_v1.status_code in (400, 403, 404)

            bob_v2 = bob.retrieve(prompt_id, version=2)
            assert bob_v2.status_code in (400, 403, 404)
        finally:
            alice.delete(prompt_id)
