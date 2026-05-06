# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
from openai import OpenAI


def get_auth_token(env_var: str, default: str) -> str:
    """Get auth token from environment variable or use default."""
    return os.environ.get(env_var, default)


@pytest.mark.integration
class TestOpenAIConversations:
    # TODO: Update to compat_client after client-SDK is generated
    def test_conversation_create(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "demo"}, items=[{"type": "message", "role": "user", "content": "Hello!"}]
        )

        assert conversation.id.startswith("conv_")
        assert conversation.object == "conversation"
        assert conversation.metadata["topic"] == "demo"
        assert isinstance(conversation.created_at, int)

    def test_conversation_retrieve(self, openai_client):
        conversation = openai_client.conversations.create(metadata={"topic": "demo"})

        retrieved = openai_client.conversations.retrieve(conversation.id)

        assert retrieved.id == conversation.id
        assert retrieved.object == "conversation"
        assert retrieved.metadata["topic"] == "demo"
        assert retrieved.created_at == conversation.created_at

    def test_conversation_update(self, openai_client):
        conversation = openai_client.conversations.create(metadata={"topic": "demo"})

        updated = openai_client.conversations.update(conversation.id, metadata={"topic": "project-x"})

        assert updated.id == conversation.id
        assert updated.metadata["topic"] == "project-x"
        assert updated.created_at == conversation.created_at

    def test_conversation_delete(self, openai_client):
        conversation = openai_client.conversations.create(metadata={"topic": "demo"})

        deleted = openai_client.conversations.delete(conversation.id)

        assert deleted.id == conversation.id
        assert deleted.object == "conversation.deleted"
        assert deleted.deleted is True

    def test_conversation_items_create(self, openai_client):
        conversation = openai_client.conversations.create()

        items = openai_client.conversations.items.create(
            conversation.id,
            items=[
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "How are you?"}]},
            ],
        )

        assert items.object == "list"
        assert len(items.data) == 2
        assert items.data[0].content[0].text == "Hello!"
        assert items.data[1].content[0].text == "How are you?"
        assert items.first_id == items.data[0].id
        assert items.last_id == items.data[1].id
        assert items.has_more is False

    def test_conversation_items_list(self, openai_client):
        conversation = openai_client.conversations.create()

        openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
        )

        items = openai_client.conversations.items.list(conversation.id, limit=10)

        assert items.object == "list"
        assert len(items.data) >= 1
        assert items.data[0].type == "message"
        assert items.data[0].role == "user"
        assert hasattr(items, "first_id")
        assert hasattr(items, "last_id")
        assert hasattr(items, "has_more")

    def test_conversation_item_retrieve(self, openai_client):
        conversation = openai_client.conversations.create()

        created_items = openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
        )

        item_id = created_items.data[0].id
        item = openai_client.conversations.items.retrieve(item_id, conversation_id=conversation.id)

        assert item.id == item_id
        assert item.type == "message"
        assert item.role == "user"
        assert item.content[0].text == "Hello!"

    def test_conversation_item_delete(self, openai_client):
        conversation = openai_client.conversations.create()

        created_items = openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
        )

        item_id = created_items.data[0].id
        deleted = openai_client.conversations.items.delete(item_id, conversation_id=conversation.id)

        assert deleted.id == item_id
        assert deleted.object == "conversation.item.deleted"
        assert deleted.deleted is True

    def test_full_workflow(self, openai_client):
        conversation = openai_client.conversations.create(
            metadata={"topic": "workflow-test"}, items=[{"type": "message", "role": "user", "content": "Hello!"}]
        )

        openai_client.conversations.items.create(
            conversation.id,
            items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Follow up"}]}],
        )

        all_items = openai_client.conversations.items.list(conversation.id)
        assert len(all_items.data) >= 2

        updated = openai_client.conversations.update(conversation.id, metadata={"topic": "workflow-complete"})
        assert updated.metadata["topic"] == "workflow-complete"

        openai_client.conversations.delete(conversation.id)


def auth_enabled() -> bool:
    """Check if auth testing is enabled via environment variable."""
    # Auth is enabled if any of the auth token env vars are set
    return bool(os.environ.get("ALICE_TOKEN") or os.environ.get("BOB_TOKEN"))


@pytest.mark.integration
@pytest.mark.skipif(not auth_enabled(), reason="Auth tokens not configured (set ALICE_TOKEN and BOB_TOKEN)")
class TestConversationAccessControl:
    """Tests for user separation and access control in conversations.

    These tests verify that:
    - Users can only access their own conversations
    - Users cannot retrieve, update, or delete other users' conversations
    - Users cannot access items in other users' conversations

    To run these tests, set ALICE_TOKEN and BOB_TOKEN environment variables
    with valid auth tokens for two different users.
    """

    @pytest.fixture
    def alice_client(self, openai_client):
        """Create an OpenAI client for Alice."""
        token = get_auth_token("ALICE_TOKEN", "token-alice")
        return OpenAI(
            base_url=str(openai_client.base_url),
            api_key=token,
            max_retries=0,
            timeout=30.0,
        )

    @pytest.fixture
    def bob_client(self, openai_client):
        """Create an OpenAI client for Bob."""
        token = get_auth_token("BOB_TOKEN", "token-bob")
        return OpenAI(
            base_url=str(openai_client.base_url),
            api_key=token,
            max_retries=0,
            timeout=30.0,
        )

    def test_user_cannot_retrieve_other_users_conversation(self, alice_client, bob_client, require_server):
        """Test that one user cannot retrieve another user's conversation."""
        # Alice creates a conversation
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice"})
        alice_conv_id = alice_conv.id

        try:
            # Alice can retrieve her own conversation
            retrieved = alice_client.conversations.retrieve(alice_conv_id)
            assert retrieved.id == alice_conv_id

            # Bob tries to retrieve Alice's conversation - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.retrieve(alice_conv_id)

            # Access should be denied - expect BadRequestError (400), NotFoundError (404), or PermissionDeniedError (403)
            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT access Alice's conversation, got status {error.status_code}: {error}"
            )
        finally:
            # Cleanup: Alice deletes her conversation
            alice_client.conversations.delete(alice_conv_id)

    def test_user_cannot_update_other_users_conversation(self, alice_client, bob_client, require_server):
        """Test that one user cannot update another user's conversation."""
        # Alice creates a conversation
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice"})
        alice_conv_id = alice_conv.id

        try:
            # Bob tries to update Alice's conversation - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.update(alice_conv_id, metadata={"hacked_by": "bob"})

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT update Alice's conversation, got status {error.status_code}: {error}"
            )

            # Verify Alice's conversation was not modified
            retrieved = alice_client.conversations.retrieve(alice_conv_id)
            assert retrieved.metadata.get("hacked_by") is None
            assert retrieved.metadata["owner"] == "alice"
        finally:
            alice_client.conversations.delete(alice_conv_id)

    def test_user_cannot_delete_other_users_conversation(self, alice_client, bob_client, require_server):
        """Test that one user cannot delete another user's conversation."""
        # Alice creates a conversation
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice"})
        alice_conv_id = alice_conv.id

        try:
            # Bob tries to delete Alice's conversation - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.delete(alice_conv_id)

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT delete Alice's conversation, got status {error.status_code}: {error}"
            )

            # Verify Alice's conversation still exists
            retrieved = alice_client.conversations.retrieve(alice_conv_id)
            assert retrieved.id == alice_conv_id
        finally:
            alice_client.conversations.delete(alice_conv_id)

    def test_user_cannot_access_other_users_conversation_items(self, alice_client, bob_client, require_server):
        """Test that one user cannot access items in another user's conversation."""
        # Alice creates a conversation and adds items
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice"})
        alice_conv_id = alice_conv.id

        try:
            # Create items and verify they were created
            created_items = alice_client.conversations.items.create(
                alice_conv_id,
                items=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello, I'm Alice!"}],
                    }
                ],
            )
            assert len(created_items.data) >= 1, "Items should be created successfully"

            # Alice can list her conversation items
            items = alice_client.conversations.items.list(alice_conv_id)
            assert len(items.data) >= 1, f"Alice should see her items, got {len(items.data)}"

            # Bob tries to list Alice's conversation items - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.items.list(alice_conv_id)

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT access Alice's items, got status {error.status_code}: {error}"
            )
        finally:
            alice_client.conversations.delete(alice_conv_id)

    def test_user_cannot_add_items_to_other_users_conversation(self, alice_client, bob_client, require_server):
        """Test that one user cannot add items to another user's conversation."""
        # Alice creates a conversation
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice"})
        alice_conv_id = alice_conv.id

        try:
            # Bob tries to add items to Alice's conversation - should fail
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.items.create(
                    alice_conv_id,
                    items=[{"type": "message", "role": "user", "content": "Injected by Bob!"}],
                )

            error = exc_info.value
            assert hasattr(error, "status_code"), f"Expected HTTP error, got: {error}"
            assert error.status_code in (400, 403, 404), (
                f"Bob should NOT add items to Alice's conversation, got status {error.status_code}: {error}"
            )

            # Verify no items were added by Bob
            items = alice_client.conversations.items.list(alice_conv_id)
            for item in items.data:
                content = getattr(item, "content", None)
                if content:
                    assert "Injected by Bob" not in str(content)
        finally:
            alice_client.conversations.delete(alice_conv_id)

    def test_users_have_isolated_conversations(self, alice_client, bob_client, require_server):
        """Test that users cannot access each other's conversations."""
        # Alice creates a conversation
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice", "test_marker": "isolation_test"})
        alice_conv_id = alice_conv.id

        # Bob creates a conversation
        bob_conv = bob_client.conversations.create(metadata={"owner": "bob", "test_marker": "isolation_test"})
        bob_conv_id = bob_conv.id

        try:
            # Bob cannot access Alice's conversation
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.retrieve(alice_conv_id)
            assert exc_info.value.status_code in (400, 403, 404)

            # Alice cannot access Bob's conversation
            with pytest.raises(Exception) as exc_info:
                alice_client.conversations.retrieve(bob_conv_id)
            assert exc_info.value.status_code in (400, 403, 404)

            # Each user can access their own
            alice_retrieved = alice_client.conversations.retrieve(alice_conv_id)
            assert alice_retrieved.id == alice_conv_id

            bob_retrieved = bob_client.conversations.retrieve(bob_conv_id)
            assert bob_retrieved.id == bob_conv_id
        finally:
            alice_client.conversations.delete(alice_conv_id)
            bob_client.conversations.delete(bob_conv_id)

    def test_user_can_access_own_resources_after_denial(self, alice_client, bob_client, require_server):
        """Test that access control doesn't interfere with legitimate access."""
        # Both users create conversations
        alice_conv = alice_client.conversations.create(metadata={"owner": "alice"})
        alice_conv_id = alice_conv.id

        bob_conv = bob_client.conversations.create(metadata={"owner": "bob"})
        bob_conv_id = bob_conv.id

        try:
            # Bob tries to access Alice's (denied)
            with pytest.raises(Exception) as exc_info:
                bob_client.conversations.retrieve(alice_conv_id)
            assert exc_info.value.status_code in (400, 403, 404)

            # Bob should still be able to access his own after being denied
            bob_retrieved = bob_client.conversations.retrieve(bob_conv_id)
            assert bob_retrieved.id == bob_conv_id
            assert bob_retrieved.metadata["owner"] == "bob"

            # Alice should still be able to access her own
            alice_retrieved = alice_client.conversations.retrieve(alice_conv_id)
            assert alice_retrieved.id == alice_conv_id
            assert alice_retrieved.metadata["owner"] == "alice"
        finally:
            alice_client.conversations.delete(alice_conv_id)
            bob_client.conversations.delete(bob_conv_id)
