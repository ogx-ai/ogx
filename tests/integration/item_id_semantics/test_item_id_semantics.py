# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Item-id semantics for the Conversations API and the Responses API.

These tests pin the reference behavior measured against live OpenAI
(api.openai.com, 2026-11-23). The reference rule is uniform: **client-supplied
item ids are never honored** — unknown well-formed ids are silently replaced
by server-minted ones, ids of ill-formed shape are rejected, and re-sending an
id that already exists (in the `previous_response_id` chain or in the target
conversation) is a 400.

Conversations API (POST /conversations/items):
  C1   no id                    -> server mints msg_ + 48 hex chars
  C2   unknown msg_ id          -> silently replaced by a server id
  C2b  non-msg-shaped id        -> 400 "Expected an ID that begins with 'msg'"
  C3   id from another conv     -> item copied with its ORIGINAL content
  C4   id already in same conv  -> 400 "Item already in conversation"

Responses API:
  R1   unknown client id in input      -> replaced (visible via input_items)
  R2   re-send id from previous_response_id chain -> 400 "Duplicate item found"
  R3   conversation= + unknown client id -> replaced; the mirror uses the new id
  R4   conversation= + re-send live conv item id  -> 400 "Duplicate item found"

Targets (see conftest.py):
  - local OGX server (default):
        pytest tests/integration/item_id_semantics --setup gpt
  - live OpenAI (reference target; no recordings needed):
        pytest tests/integration/item_id_semantics --live-openai --setup gpt
        OGX_LIVE_OPENAI=1 pytest tests/integration/item_id_semantics --text-model openai/gpt-5-nano

Known OGX divergences, pinned by these tests and marked
`@pytest.mark.divergence` (converted to a non-strict xfail on the local
target only — see conftest.py):
  D1 = C2b  — no msg_ prefix validation, arbitrary ids accepted and re-minted
  D2 = R1/R3 — client ids kept verbatim instead of replaced
  D3 = R2    — silent duplication in the stored input chain instead of 400
  D5 = R4    — silent skip in the conversation mirror instead of 400

The marked tests run in full on the local target: their expected failures
(the divergence itself, or a missing replay recording) convert to XP instead
of failures, and they report XPASS automatically once the parity fixes land,
with no test edits. On live OpenAI no xfail is applied — they must pass.
"""

import pytest
from openai import BadRequestError, OpenAI

from tests.integration.item_id_semantics.conftest import is_live_openai

HEX48_A = "ab" * 24
HEX48_B = "cd" * 24
HEX48_C = "ef" * 24
MINTED_ID_LEN = 4 + 48  # "msg_" + 48 hex chars


def item_text(item) -> str | None:
    """Extract plain text from a message item's content (string or parts)."""
    content = getattr(item, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") or "" for part in content)


def user_item(items):
    """The user-role item among conversation/response input items."""
    return next(it for it in items if getattr(it, "role", None) == "user")


def _live_model(request, text_model_id: str) -> str:
    """Model name for the target.

    The suite carries provider-prefixed ids (e.g. openai/gpt-5-nano); live
    OpenAI takes the bare model name, so the prefix is stripped there. Runs
    without a configured text model are skipped by the suite's
    skip_if_no_model fixture before the test body executes.
    """
    if is_live_openai(request.config) and text_model_id.startswith("openai/"):
        return text_model_id.split("/", 1)[1]
    return text_model_id


def _text_item(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


# --- Conversations API --------------------------------------------------------


def test_conversation_no_id_server_mints(id_sem_client: OpenAI, conversation_registry: list[str]):
    conv = id_sem_client.conversations.create()
    conversation_registry.append(conv.id)
    items = id_sem_client.conversations.items.create(conv.id, items=[_text_item("hello")])
    assert len(items.data) == 1
    item = items.data[0]
    assert item.id
    assert item.id.startswith("msg_")
    assert len(item.id) == MINTED_ID_LEN
    assert item_text(item) == "hello"


def test_conversation_unknown_id_replaced(id_sem_client: OpenAI, conversation_registry: list[str]):
    sent = f"msg_{HEX48_A}"
    conv = id_sem_client.conversations.create()
    conversation_registry.append(conv.id)
    items = id_sem_client.conversations.items.create(
        conv.id, items=[{**_text_item("sent with unknown id"), "id": sent}]
    )
    assert len(items.data) == 1
    item = items.data[0]
    assert item.id != sent
    assert item.id.startswith("msg_")
    assert len(item.id) == MINTED_ID_LEN
    assert item_text(item) == "sent with unknown id"


@pytest.mark.divergence("D1", "OGX accepts arbitrary client ids and re-mints them instead of 400ing on shape")
def test_conversation_non_msg_id_rejected(id_sem_client: OpenAI, conversation_registry: list[str]):
    conv = id_sem_client.conversations.create()
    conversation_registry.append(conv.id)
    with pytest.raises(BadRequestError) as exc:
        id_sem_client.conversations.items.create(
            conv.id, items=[{**_text_item("sent with arbitrary id"), "id": "my-arbitrary-id-123"}]
        )
    assert "begins with 'msg'" in exc.value.message


def test_conversation_id_copied_from_other_conversation(id_sem_client: OpenAI, conversation_registry: list[str]):
    src = id_sem_client.conversations.create()
    conversation_registry.append(src.id)
    created = id_sem_client.conversations.items.create(src.id, items=[_text_item("hello")])
    src_id = created.data[0].id

    dst = id_sem_client.conversations.create()
    conversation_registry.append(dst.id)
    items = id_sem_client.conversations.items.create(dst.id, items=[{**_text_item("content sent for B"), "id": src_id}])
    assert len(items.data) == 1
    item = items.data[0]
    assert item.id == src_id
    # the source item's original content wins over the re-sent content
    assert item_text(item) == "hello"


def test_conversation_duplicate_id_same_conversation_rejected(id_sem_client: OpenAI, conversation_registry: list[str]):
    conv = id_sem_client.conversations.create()
    conversation_registry.append(conv.id)
    created = id_sem_client.conversations.items.create(conv.id, items=[_text_item("hello")])
    item_id = created.data[0].id

    with pytest.raises(BadRequestError) as exc:
        id_sem_client.conversations.items.create(
            conv.id, items=[{**_text_item("attempted replacement"), "id": item_id}]
        )
    assert "Item already in conversation" in exc.value.message
    # the stored item is unchanged
    stored = [it for it in id_sem_client.conversations.items.list(conv.id).data if it.id == item_id]
    assert len(stored) == 1
    assert item_text(stored[0]) == "hello"


# --- Responses API --------------------------------------------------------------


@pytest.mark.divergence(
    "D2", "OGX keeps client-supplied ids verbatim in the stored response input instead of replacing them"
)
def test_responses_unknown_input_id_replaced(request, id_sem_client: OpenAI, text_model_id: str):
    sent = f"msg_{HEX48_B}"
    resp = id_sem_client.responses.create(
        model=_live_model(request, text_model_id), input=[{"role": "user", "id": sent, "content": "hi"}]
    )
    item = user_item(id_sem_client.responses.input_items.list(resp.id).data)
    assert item.id != sent
    assert item.id.startswith("msg_")
    assert item_text(item) == "hi"


@pytest.mark.divergence("D3", "OGX silently duplicates the item in the stored input chain instead of 400ing")
def test_responses_resend_id_from_previous_response_rejected(request, id_sem_client: OpenAI, text_model_id: str):
    model = _live_model(request, text_model_id)
    resp1 = id_sem_client.responses.create(
        model=model,
        input=[{"role": "user", "content": "My favorite color is GREEN. Do not state the color, just acknowledge."}],
    )
    prior = user_item(id_sem_client.responses.input_items.list(resp1.id).data)
    assert prior.id

    with pytest.raises(BadRequestError) as exc:
        id_sem_client.responses.create(
            model=model,
            previous_response_id=resp1.id,
            input=[
                {"role": "user", "id": prior.id, "content": "My favorite color is BLUE."},
                {"role": "user", "content": "What is my favorite color, according to my message?"},
            ],
        )
    assert "Duplicate item found with id" in exc.value.message


@pytest.mark.divergence(
    "D2",
    "OGX keeps client-supplied ids verbatim (in the response input and the conversation mirror) instead of replacing them",
)
def test_responses_conversation_unknown_input_id_replaced(
    request, id_sem_client: OpenAI, conversation_registry: list[str], text_model_id: str
):
    sent = f"msg_{HEX48_C}"
    conv = id_sem_client.conversations.create()
    conversation_registry.append(conv.id)
    resp = id_sem_client.responses.create(
        model=_live_model(request, text_model_id),
        conversation=conv.id,
        input=[{"role": "user", "id": sent, "content": "hi"}],
    )
    resp_item = user_item(id_sem_client.responses.input_items.list(resp.id).data)
    assert resp_item.id != sent
    assert resp_item.id.startswith("msg_")
    # the conversation mirror holds the replaced id, not the client's
    conv_users = [
        it for it in id_sem_client.conversations.items.list(conv.id).data if getattr(it, "role", None) == "user"
    ]
    assert len(conv_users) == 1
    assert conv_users[0].id == resp_item.id


@pytest.mark.divergence("D5", "OGX silently skips the duplicate in the conversation mirror instead of 400ing")
def test_responses_conversation_resend_existing_item_id_rejected(
    request, id_sem_client: OpenAI, conversation_registry: list[str], text_model_id: str
):
    model = _live_model(request, text_model_id)
    conv = id_sem_client.conversations.create()
    conversation_registry.append(conv.id)
    id_sem_client.responses.create(model=model, conversation=conv.id, input=[{"role": "user", "content": "hi"}])
    live = user_item(id_sem_client.conversations.items.list(conv.id).data)

    with pytest.raises(BadRequestError) as exc:
        id_sem_client.responses.create(
            model=model,
            conversation=conv.id,
            input=[
                {"role": "user", "id": live.id, "content": "resent via responses"},
                {
                    "role": "user",
                    "content": "and a follow-up question: how many messages have I sent in this conversation?",
                },
            ],
        )
    assert "Duplicate item found with id" in exc.value.message
