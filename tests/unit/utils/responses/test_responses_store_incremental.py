# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
import time
from tempfile import TemporaryDirectory
from uuid import uuid4

from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import (
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseObject,
    OpenAIUserMessageParam,
    Order,
)


def build_store(db_path: str, policy: list | None = None) -> ResponsesStore:
    backend_name = f"sql_responses_{uuid4().hex}"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path)})
    return ResponsesStore(
        ResponsesStoreReference(backend=backend_name, table_name="responses"),
        policy=policy or [],
    )


def create_test_response_input(content: str, input_id: str) -> OpenAIResponseInput:
    """Helper to create a test response input."""
    from ogx_api import OpenAIResponseMessage

    return OpenAIResponseMessage(
        id=input_id,
        content=content,
        role="user",
        type="message",
    )


def create_test_messages(content: str) -> list[OpenAIMessageParam]:
    """Helper to create test messages for chat completion."""
    return [OpenAIUserMessageParam(content=content)]


async def test_responses_store_incremental_input_reconstruction():
    """Test that responses stored with incremental_input=True reconstruct the full
    input chain when read via get_response_object."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        # Turn 1: no previous_response_id, stored normally
        resp1 = OpenAIResponseObject(
            id="resp-1",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        input1 = [create_test_response_input("Question 1", "input-1")]
        await store.store_response_object(resp1, input1, create_test_messages("Question 1"))

        # Turn 2: references resp-1, stored incrementally (only new input)
        resp2 = OpenAIResponseObject(
            id="resp-2",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-2",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 2")],
                )
            ],
            status="completed",
            previous_response_id="resp-1",
            store=True,
        )
        input2 = [create_test_response_input("Question 2", "input-2")]
        await store.store_response_object(resp2, input2, create_test_messages("Question 2"), incremental_input=True)

        # Turn 3: references resp-2, stored incrementally
        resp3 = OpenAIResponseObject(
            id="resp-3",
            created_at=base_time + 2,
            model="test-model",
            object="response",
            output=[],
            status="completed",
            previous_response_id="resp-2",
            store=True,
        )
        input3 = [create_test_response_input("Question 3", "input-3")]
        await store.store_response_object(resp3, input3, create_test_messages("Question 3"), incremental_input=True)

        await store.flush()

        # Reading resp-1 should return just its own input (no chain)
        r1 = await store.get_response_object("resp-1")
        assert len(r1.input) == 1
        assert r1.input[0].content == "Question 1"

        # Reading resp-2 should reconstruct: resp-1.input + resp-1.output + resp-2.input
        r2 = await store.get_response_object("resp-2")
        assert len(r2.input) == 3
        assert r2.input[0].content == "Question 1"
        assert r2.input[1].content[0].text == "Answer 1"
        assert r2.input[2].content == "Question 2"

        # Reading resp-3 should reconstruct the full chain:
        # resp-1.input + resp-1.output + resp-2.input + resp-2.output + resp-3.input
        r3 = await store.get_response_object("resp-3")
        assert len(r3.input) == 5
        assert r3.input[0].content == "Question 1"
        assert r3.input[1].content[0].text == "Answer 1"
        assert r3.input[2].content == "Question 2"
        assert r3.input[3].content[0].text == "Answer 2"
        assert r3.input[4].content == "Question 3"

        # Reading with reconstruct_input=False should return only the stored new input
        r3_raw = await store.get_response_object("resp-3", reconstruct_input=False)
        assert len(r3_raw.input) == 1
        assert r3_raw.input[0].content == "Question 3"

        # list_response_input_items should also use the reconstructed input
        items = await store.list_response_input_items("resp-3", order=Order.asc)
        assert len(items.data) == 5


async def test_responses_store_incremental_input_backward_compat():
    """Test that responses stored in the old full-accumulation format still work
    correctly when mixed with new incremental format in a chain."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        # Turn 1: old format (no incremental flag)
        resp1 = OpenAIResponseObject(
            id="resp-old-1",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-old-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Old answer")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            resp1,
            [create_test_response_input("Old question", "input-old-1")],
            create_test_messages("Old question"),
        )

        # Turn 2: old format with previous_response_id (full accumulated input)
        resp2 = OpenAIResponseObject(
            id="resp-old-2",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-old-2",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Old answer 2")],
                )
            ],
            status="completed",
            previous_response_id="resp-old-1",
            store=True,
        )
        # Old format: stored full accumulated input
        await store.store_response_object(
            resp2,
            [
                create_test_response_input("Old question", "input-old-1"),
                OpenAIResponseMessage(
                    id="out-old-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Old answer")],
                ),
                create_test_response_input("Second question", "input-old-2"),
            ],
            create_test_messages("Second question"),
            incremental_input=False,
        )

        # Turn 3: NEW format referencing old-format response
        resp3 = OpenAIResponseObject(
            id="resp-new-3",
            created_at=base_time + 2,
            model="test-model",
            object="response",
            output=[],
            status="completed",
            previous_response_id="resp-old-2",
            store=True,
        )
        await store.store_response_object(
            resp3,
            [create_test_response_input("New question", "input-new-3")],
            create_test_messages("New question"),
            incremental_input=True,
        )

        await store.flush()

        # resp-old-2 has no incremental flag, so get_response_object returns its stored input as-is
        r2 = await store.get_response_object("resp-old-2")
        assert len(r2.input) == 3

        # resp-new-3 is incremental, so it reconstructs from resp-old-2 (which is NOT incremental)
        # Reconstruction stops at resp-old-2 since it already has full accumulated input
        r3 = await store.get_response_object("resp-new-3")
        # resp-old-2.input (3 items) + resp-old-2.output (1 item) + resp-new-3.input (1 item) = 5
        assert len(r3.input) == 5
        assert r3.input[0].content == "Old question"
        assert r3.input[3].content[0].text == "Old answer 2"
        assert r3.input[4].content == "New question"


async def test_responses_store_incremental_input_reconstruction_handles_deep_chains():
    """Deep ancestry chains should reconstruct without recursion-limit failures."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())
        total_turns = 250
        previous_response_id: str | None = None

        for i in range(total_turns):
            response_id = f"resp-deep-{i}"
            response = OpenAIResponseObject(
                id=response_id,
                created_at=base_time + i,
                model="test-model",
                object="response",
                output=[
                    OpenAIResponseMessage(
                        id=f"out-deep-{i}",
                        role="assistant",
                        content=[OpenAIResponseOutputMessageContentOutputText(text=f"Answer {i}")],
                    )
                ],
                status="completed",
                previous_response_id=previous_response_id,
                store=True,
            )

            await store.store_response_object(
                response,
                [create_test_response_input(f"Question {i}", f"input-deep-{i}")],
                create_test_messages(f"Question {i}"),
                incremental_input=previous_response_id is not None,
            )
            previous_response_id = response_id

        assert previous_response_id is not None

        original_recursion_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(200)
            reconstructed = await store.get_response_object(previous_response_id)
        finally:
            sys.setrecursionlimit(original_recursion_limit)

        assert len(reconstructed.input) == (2 * total_turns - 1)
        assert reconstructed.input[0].content == "Question 0"
        assert reconstructed.input[-1].content == f"Question {total_turns - 1}"


async def test_update_response_object_preserves_incremental_storage_mode():
    """Updating an incremental response should not disable input reconstruction."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        resp1 = OpenAIResponseObject(
            id="resp-upd-1",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-upd-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            resp1,
            [create_test_response_input("Question 1", "input-upd-1")],
            create_test_messages("Question 1"),
        )

        resp2 = OpenAIResponseObject(
            id="resp-upd-2",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[],
            status="in_progress",
            previous_response_id="resp-upd-1",
            store=True,
        )
        await store.store_response_object(
            resp2,
            [create_test_response_input("Question 2", "input-upd-2")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        before_update = await store.get_response_object("resp-upd-2")
        assert len(before_update.input) == 3

        raw = await store.get_response_object("resp-upd-2", reconstruct_input=False)
        raw.status = "completed"
        await store.update_response_object(raw)

        after_update = await store.get_response_object("resp-upd-2")
        assert len(after_update.input) == 3
        assert after_update.input[0].content == "Question 1"
        assert after_update.input[1].content[0].text == "Answer 1"
        assert after_update.input[2].content == "Question 2"


async def test_list_responses_reconstructs_incremental_input():
    """list_responses should return full reconstructed input for incremental rows."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        resp1 = OpenAIResponseObject(
            id="resp-list-1",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-list-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            resp1,
            [create_test_response_input("Question 1", "input-list-1")],
            create_test_messages("Question 1"),
        )

        resp2 = OpenAIResponseObject(
            id="resp-list-2",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[],
            status="completed",
            previous_response_id="resp-list-1",
            store=True,
        )
        await store.store_response_object(
            resp2,
            [create_test_response_input("Question 2", "input-list-2")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        listed = await store.list_responses(limit=10, order=Order.desc)
        listed_resp2 = next(item for item in listed.data if item.id == "resp-list-2")

        assert len(listed_resp2.input) == 3
        assert listed_resp2.input[0].content == "Question 1"
        assert listed_resp2.input[1].content[0].text == "Answer 1"
        assert listed_resp2.input[2].content == "Question 2"
