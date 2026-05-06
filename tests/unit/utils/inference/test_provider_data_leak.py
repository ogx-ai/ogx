# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Reproduces the PROVIDER_DATA_VAR contextvar leak through background worker tasks.

The inference store uses a write queue with long-lived worker tasks (on Postgres).
asyncio.create_task copies all contextvars at creation time, so the worker
permanently inherits the first request's PROVIDER_DATA_VAR. This means every
DB write is stamped with the first user's identity, regardless of who actually
made the request.

This test forces the write queue on (normally disabled for SQLite) to demonstrate
the leak without needing a Postgres instance.
"""

import time

import pytest

from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.datatypes import User
from ogx.core.request_headers import PROVIDER_DATA_VAR
from ogx.core.storage.datatypes import InferenceStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIUserMessageParam,
)


@pytest.fixture(autouse=True)
def setup_backends(tmp_path):
    db_path = str(tmp_path / "test_leak.db")
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=db_path)})


def _set_authenticated_user(user: User | None):
    """Simulate what ProviderDataMiddleware does for each request."""
    if user:
        PROVIDER_DATA_VAR.set({"__authenticated_user": user})
    else:
        PROVIDER_DATA_VAR.set(None)


def _make_completion(completion_id: str, created: int) -> OpenAIChatCompletion:
    return OpenAIChatCompletion(
        id=completion_id,
        created=created,
        model="test-model",
        object="chat.completion",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content=f"Response for {completion_id}",
                ),
                finish_reason="stop",
            )
        ],
    )


async def test_provider_data_leak_through_write_queue():
    """Demonstrates that PROVIDER_DATA_VAR leaks into background workers.

    Expected behavior: each completion should be owned by the user who created it.
    Actual behavior: all completions are owned by whoever triggered worker creation.
    """
    owner_policy = [
        AccessRule(permit=Scope(actions=[Action.READ]), when=["user is owner"]),
        AccessRule(permit=Scope(actions=[Action.CREATE]), when=[]),
    ]

    reference = InferenceStoreReference(
        backend="sql_default",
        table_name="leak_test",
        num_writers=1,
    )
    store = InferenceStore(reference, policy=owner_policy)
    await store.initialize()

    # Force the write queue on (normally disabled for SQLite)
    store.enable_write_queue = True

    alice = User(principal="alice", attributes={"roles": ["user"]})
    bob = User(principal="bob", attributes={"roles": ["user"]})

    base_time = int(time.time())

    # --- Request 1: Alice creates a completion ---
    # This is the first request, so it spawns the background worker.
    # The worker inherits Alice's PROVIDER_DATA_VAR permanently.
    _set_authenticated_user(alice)
    await store.store_chat_completion(
        _make_completion("alice-completion", base_time + 1),
        [OpenAIUserMessageParam(role="user", content="Hello from Alice")],
    )
    await store.flush()

    # --- Request 2: Bob creates a completion ---
    # The worker is already running with Alice's context.
    # Bob's write goes through the queue but is processed under Alice's identity.
    _set_authenticated_user(bob)
    await store.store_chat_completion(
        _make_completion("bob-completion", base_time + 2),
        [OpenAIUserMessageParam(role="user", content="Hello from Bob")],
    )
    await store.flush()

    # --- Now verify: can each user see only their own completions? ---

    # Alice should see 1 completion (her own)
    _set_authenticated_user(alice)
    alice_results = await store.list_chat_completions()

    # Bob should see 1 completion (his own)
    _set_authenticated_user(bob)
    bob_results = await store.list_chat_completions()

    await store.shutdown()

    # --- Assertions ---
    alice_ids = [c.id for c in alice_results.data]
    bob_ids = [c.id for c in bob_results.data]

    print(f"\nAlice sees: {alice_ids}")
    print(f"Bob sees:   {bob_ids}")

    # If the bug exists:
    #   Alice sees: ['alice-completion', 'bob-completion']  (both!)
    #   Bob sees:   []  (nothing!)
    #
    # If fixed:
    #   Alice sees: ['alice-completion']
    #   Bob sees:   ['bob-completion']

    assert "alice-completion" in alice_ids, "Alice should see her own completion"
    assert "bob-completion" not in alice_ids, (
        "BUG: Alice can see Bob's completion — PROVIDER_DATA_VAR leaked from worker"
    )
    assert "bob-completion" in bob_ids, "Bob should see his own completion"
    assert "alice-completion" not in bob_ids, "BUG: Bob can see Alice's completion — unexpected cross-contamination"
