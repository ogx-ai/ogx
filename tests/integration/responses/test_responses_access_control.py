# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for user separation and access control in the Responses API.

These tests verify that:
- Users can only access their own stored responses
- Users cannot retrieve or delete other users' responses
- Users cannot list other users' responses
"""

import json
import os

import pytest
from openai import OpenAI


def get_auth_token(env_var: str, default: str) -> str:
    """Get auth token from environment variable or use default."""
    return os.environ.get(env_var, default)


def auth_enabled() -> bool:
    """Check if auth testing is enabled via environment variable."""
    # Auth is enabled if any of the auth token env vars are set
    return bool(os.environ.get("ALICE_TOKEN") or os.environ.get("BOB_TOKEN"))


@pytest.mark.integration
@pytest.mark.skipif(not auth_enabled(), reason="Auth tokens not configured (set ALICE_TOKEN and BOB_TOKEN)")
class TestResponsesAccessControl:
    """Tests for user separation and access control in stored responses.

    These tests verify that:
    - Users can only access their own stored responses
    - Users cannot retrieve or delete other users' responses
    - Response listings only show the user's own responses

    To run these tests, set ALICE_TOKEN and BOB_TOKEN environment variables
    with valid auth tokens for two different users.
    """

    @pytest.fixture
    def alice_client(self, openai_client, request):
        """Create an OpenAI client for Alice."""
        token = get_auth_token("ALICE_TOKEN", "token-alice")
        # Send test id so server uses correct recordings dir in replay mode
        default_headers = {
            "X-OGX-Provider-Data": json.dumps({"__test_id": request.node.nodeid}),
        }
        return OpenAI(
            base_url=str(openai_client.base_url),
            api_key=token,
            default_headers=default_headers,
            max_retries=0,
            timeout=60.0,
        )

    @pytest.fixture
    def bob_client(self, openai_client, request):
        """Create an OpenAI client for Bob."""
        token = get_auth_token("BOB_TOKEN", "token-bob")
        default_headers = {
            "X-OGX-Provider-Data": json.dumps({"__test_id": request.node.nodeid}),
        }
        return OpenAI(
            base_url=str(openai_client.base_url),
            api_key=token,
            default_headers=default_headers,
            max_retries=0,
            timeout=60.0,
        )

    def _create_stored_response(self, client, text_model_id: str, input_text: str = "Say hello"):
        """Helper to create a stored response."""
        response = client.responses.create(
            model=text_model_id,
            input=input_text,
            store=True,
        )
        return response

    def test_user_cannot_retrieve_other_users_response(self, alice_client, bob_client, text_model_id, require_server):
        """Test that one user cannot retrieve another user's stored response."""
        # Alice creates a stored response
        alice_response = self._create_stored_response(alice_client, text_model_id, "Hello from Alice")
        alice_response_id = alice_response.id

        try:
            # Alice can retrieve her own response
            retrieved = alice_client.responses.retrieve(alice_response_id)
            assert retrieved.id == alice_response_id

            # Bob tries to retrieve Alice's response - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.responses.retrieve(alice_response_id)

            # Access should be denied - expect 400, 403, or 404
            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT access Alice's response, got status {error.status_code}: {error}"
            )
        finally:
            # Cleanup: Alice deletes her response
            try:
                alice_client.responses.delete(alice_response_id)
            except Exception:
                pass  # Ignore cleanup errors

    def test_user_cannot_delete_other_users_response(self, alice_client, bob_client, text_model_id, require_server):
        """Test that one user cannot delete another user's stored response."""
        # Alice creates a stored response
        alice_response = self._create_stored_response(alice_client, text_model_id, "Hello from Alice")
        alice_response_id = alice_response.id

        try:
            # Bob tries to delete Alice's response - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.responses.delete(alice_response_id)

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT delete Alice's response, got status {error.status_code}: {error}"
            )

            # Verify Alice's response still exists
            retrieved = alice_client.responses.retrieve(alice_response_id)
            assert retrieved.id == alice_response_id
        finally:
            # Cleanup: Alice deletes her response
            try:
                alice_client.responses.delete(alice_response_id)
            except Exception:
                pass

    def test_users_have_isolated_responses(self, alice_client, bob_client, text_model_id, require_server):
        """Test that users cannot access each other's responses."""
        # Alice creates a response
        alice_response = self._create_stored_response(alice_client, text_model_id, "Alice's secret")
        alice_response_id = alice_response.id

        # Bob creates a response
        bob_response = self._create_stored_response(bob_client, text_model_id, "Bob's secret")
        bob_response_id = bob_response.id

        try:
            # Bob cannot access Alice's response
            with pytest.raises(Exception) as exc_info:
                bob_client.responses.retrieve(alice_response_id)
            assert exc_info.value.status_code in (400, 403, 404)

            # Alice cannot access Bob's response
            with pytest.raises(Exception) as exc_info:
                alice_client.responses.retrieve(bob_response_id)
            assert exc_info.value.status_code in (400, 403, 404)

            # Each user can access their own
            alice_retrieved = alice_client.responses.retrieve(alice_response_id)
            assert alice_retrieved.id == alice_response_id

            bob_retrieved = bob_client.responses.retrieve(bob_response_id)
            assert bob_retrieved.id == bob_response_id
        finally:
            try:
                alice_client.responses.delete(alice_response_id)
            except Exception:
                pass
            try:
                bob_client.responses.delete(bob_response_id)
            except Exception:
                pass

    def test_user_can_access_own_resources_after_denial(self, alice_client, bob_client, text_model_id, require_server):
        """Test that access control doesn't interfere with legitimate access."""
        # Both users create responses
        alice_response = self._create_stored_response(alice_client, text_model_id, "Alice's data")
        alice_response_id = alice_response.id

        bob_response = self._create_stored_response(bob_client, text_model_id, "Bob's data")
        bob_response_id = bob_response.id

        try:
            # Bob tries to access Alice's (denied)
            with pytest.raises(Exception) as exc_info:
                bob_client.responses.retrieve(alice_response_id)
            assert exc_info.value.status_code in (400, 403, 404)

            # Bob should still be able to access his own after being denied
            bob_retrieved = bob_client.responses.retrieve(bob_response_id)
            assert bob_retrieved.id == bob_response_id

            # Alice should still be able to access her own
            alice_retrieved = alice_client.responses.retrieve(alice_response_id)
            assert alice_retrieved.id == alice_response_id
        finally:
            try:
                alice_client.responses.delete(alice_response_id)
            except Exception:
                pass
            try:
                bob_client.responses.delete(bob_response_id)
            except Exception:
                pass

    def test_user_cannot_access_other_users_response_input_items(
        self, alice_client, bob_client, text_model_id, require_server
    ):
        """Test that one user cannot access input items from another user's response."""
        # Alice creates a stored response with input
        alice_response = self._create_stored_response(alice_client, text_model_id, "Hello from Alice")
        alice_response_id = alice_response.id

        try:
            # Alice can list her response's input items
            alice_items = alice_client.responses.input_items.list(alice_response_id)
            assert len(alice_items.data) >= 1, "Alice should see her input items"

            # Bob tries to list Alice's response input items - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.responses.input_items.list(alice_response_id)

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT access Alice's input items, got status {error.status_code}: {error}"
            )
        finally:
            try:
                alice_client.responses.delete(alice_response_id)
            except Exception:
                pass

    def test_previous_response_id_access_control(self, alice_client, bob_client, text_model_id, require_server):
        """Test that users cannot use another user's response as previous_response_id."""
        # Alice creates a stored response
        alice_response = self._create_stored_response(alice_client, text_model_id, "Initial message from Alice")
        alice_response_id = alice_response.id

        try:
            # Alice can use her own previous_response_id
            alice_followup = alice_client.responses.create(
                model=text_model_id,
                input="Continue the conversation",
                previous_response_id=alice_response_id,
                store=True,
            )
            assert alice_followup.id is not None

            # Bob tries to use Alice's response as previous_response_id - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.responses.create(
                    model=text_model_id,
                    input="Trying to hijack Alice's conversation",
                    previous_response_id=alice_response_id,
                    store=True,
                )

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT use Alice's previous_response_id, got status {error.status_code}: {error}"
            )
        finally:
            try:
                alice_client.responses.delete(alice_response_id)
            except Exception:
                pass
