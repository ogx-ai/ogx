# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.datatypes import User
from ogx.core.request_headers import RequestProviderDataContext
from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx_api import (
    OpenAIMessageParam,
    OpenAIResponseInput,
    OpenAIResponseObject,
    OpenAIUserMessageParam,
    Order,
    ResponseNotFoundError,
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


async def test_delete_response_preserves_descendant_incremental_input_chain():
    """Deleting an ancestor should not truncate descendant reconstructed input."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        resp1 = OpenAIResponseObject(
            id="resp-del-1",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-del-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            resp1,
            [create_test_response_input("Question 1", "input-del-1")],
            create_test_messages("Question 1"),
        )

        resp2 = OpenAIResponseObject(
            id="resp-del-2",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-del-2",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 2")],
                )
            ],
            status="completed",
            previous_response_id="resp-del-1",
            store=True,
        )
        await store.store_response_object(
            resp2,
            [create_test_response_input("Question 2", "input-del-2")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        resp3 = OpenAIResponseObject(
            id="resp-del-3",
            created_at=base_time + 2,
            model="test-model",
            object="response",
            output=[],
            status="completed",
            previous_response_id="resp-del-2",
            store=True,
        )
        await store.store_response_object(
            resp3,
            [create_test_response_input("Question 3", "input-del-3")],
            create_test_messages("Question 3"),
            incremental_input=True,
        )

        before_delete = await store.get_response_object("resp-del-3")
        assert len(before_delete.input) == 5

        await store.delete_response_object("resp-del-1")

        # Direct child should have been materialized to full input snapshot.
        resp2_raw_after_delete = await store.get_response_object("resp-del-2", reconstruct_input=False)
        assert len(resp2_raw_after_delete.input) == 3
        assert resp2_raw_after_delete.input[0].content == "Question 1"
        assert resp2_raw_after_delete.input[1].content[0].text == "Answer 1"
        assert resp2_raw_after_delete.input[2].content == "Question 2"

        # Descendant reconstruction should remain stable after ancestor deletion.
        after_delete = await store.get_response_object("resp-del-3")
        assert len(after_delete.input) == 5
        assert after_delete.input[0].content == "Question 1"
        assert after_delete.input[1].content[0].text == "Answer 1"
        assert after_delete.input[2].content == "Question 2"
        assert after_delete.input[3].content[0].text == "Answer 2"
        assert after_delete.input[4].content == "Question 3"


async def test_delete_response_materialization_does_not_require_update_permission():
    """Deleting a response should not require UPDATE permission on incremental children."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        owner_policy = [
            AccessRule(permit=Scope(actions=[Action.CREATE, Action.READ, Action.DELETE]), when=["user is owner"])
        ]
        store = build_store(db_path, policy=owner_policy)
        await store.initialize()

        user = User(principal="alice", attributes={"roles": ["member"]})
        base_time = int(time.time())

        with RequestProviderDataContext(user=user):
            parent = OpenAIResponseObject(
                id="resp-policy-parent",
                created_at=base_time,
                model="test-model",
                object="response",
                output=[
                    OpenAIResponseMessage(
                        id="out-policy-parent",
                        role="assistant",
                        content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                    )
                ],
                status="completed",
                store=True,
            )
            await store.store_response_object(
                parent,
                [create_test_response_input("Question 1", "input-policy-parent")],
                create_test_messages("Question 1"),
            )

            child = OpenAIResponseObject(
                id="resp-policy-child",
                created_at=base_time + 1,
                model="test-model",
                object="response",
                output=[],
                status="completed",
                previous_response_id="resp-policy-parent",
                store=True,
            )
            await store.store_response_object(
                child,
                [create_test_response_input("Question 2", "input-policy-child")],
                create_test_messages("Question 2"),
                incremental_input=True,
            )

            await store.delete_response_object("resp-policy-parent")

            child_after_delete = await store.get_response_object("resp-policy-child", reconstruct_input=False)
            assert len(child_after_delete.input) == 3
            assert child_after_delete.input[0].content == "Question 1"
            assert child_after_delete.input[1].content[0].text == "Answer 1"
            assert child_after_delete.input[2].content == "Question 2"

            with pytest.raises(ResponseNotFoundError):
                await store.get_response_object("resp-policy-parent")


async def test_delete_response_materializes_children_hidden_by_read_policy():
    """Deleting a readable parent should still materialize children hidden by READ policy."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path, policy=[])
        await store.initialize()

        base_time = int(time.time())
        parent = OpenAIResponseObject(
            id="resp-hidden-parent",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-hidden-parent",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            parent,
            [create_test_response_input("Question 1", "input-hidden-parent")],
            create_test_messages("Question 1"),
        )

        bob = User(principal="bob", attributes={"roles": ["bob-role"]})
        with RequestProviderDataContext(user=bob):
            child = OpenAIResponseObject(
                id="resp-hidden-child",
                created_at=base_time + 1,
                model="test-model",
                object="response",
                output=[],
                status="completed",
                previous_response_id="resp-hidden-parent",
                store=True,
            )
            await store.store_response_object(
                child,
                [create_test_response_input("Question 2", "input-hidden-child")],
                create_test_messages("Question 2"),
                incremental_input=True,
            )
            child_before_delete = await store.get_response_object("resp-hidden-child")
            assert len(child_before_delete.input) == 3

        alice = User(principal="alice", attributes={"roles": ["alice-role"]})
        with RequestProviderDataContext(user=alice):
            await store.delete_response_object("resp-hidden-parent")
            with pytest.raises(ResponseNotFoundError):
                await store.get_response_object("resp-hidden-child")

        with RequestProviderDataContext(user=bob):
            child_raw_after_delete = await store.get_response_object("resp-hidden-child", reconstruct_input=False)
            assert child_raw_after_delete.input_storage_mode is None
            assert child_raw_after_delete.previous_response_id is None
            assert len(child_raw_after_delete.input) == 3
            assert child_raw_after_delete.input[0].content == "Question 1"
            assert child_raw_after_delete.input[1].content[0].text == "Answer 1"
            assert child_raw_after_delete.input[2].content == "Question 2"

            child_after_delete = await store.get_response_object("resp-hidden-child")
            assert len(child_after_delete.input) == 3
            assert child_after_delete.input[0].content == "Question 1"
            assert child_after_delete.input[1].content[0].text == "Answer 1"
            assert child_after_delete.input[2].content == "Question 2"


async def test_delete_response_rewrites_child_ancestry_without_cross_principal_truncation():
    """Deleting by a narrower reader should not permanently truncate broader readers."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        role_policy = [
            AccessRule(permit=Scope(actions=[Action.CREATE, Action.READ, Action.DELETE]), when=["user in owners roles"])
        ]
        store = build_store(db_path, policy=role_policy)
        await store.initialize()

        base_time = int(time.time())
        user_r1 = User(principal="shared-user", attributes={"roles": ["r1"]})
        user_r1_r2 = User(principal="shared-user", attributes={"roles": ["r1", "r2"]})
        user_r2 = User(principal="shared-user", attributes={"roles": ["r2"]})

        with RequestProviderDataContext(user=user_r1):
            grandparent = OpenAIResponseObject(
                id="resp-shared-grandparent",
                created_at=base_time,
                model="test-model",
                object="response",
                output=[
                    OpenAIResponseMessage(
                        id="out-shared-grandparent",
                        role="assistant",
                        content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                    )
                ],
                status="completed",
                store=True,
            )
            await store.store_response_object(
                grandparent,
                [create_test_response_input("Question 1", "input-shared-grandparent")],
                create_test_messages("Question 1"),
            )

        with RequestProviderDataContext(user=user_r1_r2):
            parent = OpenAIResponseObject(
                id="resp-shared-parent",
                created_at=base_time + 1,
                model="test-model",
                object="response",
                output=[
                    OpenAIResponseMessage(
                        id="out-shared-parent",
                        role="assistant",
                        content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 2")],
                    )
                ],
                status="completed",
                previous_response_id="resp-shared-grandparent",
                store=True,
            )
            await store.store_response_object(
                parent,
                [create_test_response_input("Question 2", "input-shared-parent")],
                create_test_messages("Question 2"),
                incremental_input=True,
            )

            child = OpenAIResponseObject(
                id="resp-shared-child",
                created_at=base_time + 2,
                model="test-model",
                object="response",
                output=[],
                status="completed",
                previous_response_id="resp-shared-parent",
                store=True,
            )
            await store.store_response_object(
                child,
                [create_test_response_input("Question 3", "input-shared-child")],
                create_test_messages("Question 3"),
                incremental_input=True,
            )

        with RequestProviderDataContext(user=user_r1):
            child_for_r1_before_delete = await store.get_response_object("resp-shared-child")
            assert len(child_for_r1_before_delete.input) == 5
            assert child_for_r1_before_delete.input[0].content == "Question 1"
            assert child_for_r1_before_delete.input[1].content[0].text == "Answer 1"
            assert child_for_r1_before_delete.input[2].content == "Question 2"
            assert child_for_r1_before_delete.input[3].content[0].text == "Answer 2"
            assert child_for_r1_before_delete.input[4].content == "Question 3"

        with RequestProviderDataContext(user=user_r2):
            child_for_r2_before_delete = await store.get_response_object("resp-shared-child")
            assert len(child_for_r2_before_delete.input) == 3
            assert child_for_r2_before_delete.input[0].content == "Question 2"
            assert child_for_r2_before_delete.input[1].content[0].text == "Answer 2"
            assert child_for_r2_before_delete.input[2].content == "Question 3"

            await store.delete_response_object("resp-shared-parent")

            child_raw_after_delete = await store.get_response_object("resp-shared-child", reconstruct_input=False)
            assert child_raw_after_delete.input_storage_mode == "incremental"
            assert child_raw_after_delete.previous_response_id == "resp-shared-grandparent"
            assert len(child_raw_after_delete.input) == 3
            assert child_raw_after_delete.input[0].content == "Question 2"
            assert child_raw_after_delete.input[1].content[0].text == "Answer 2"
            assert child_raw_after_delete.input[2].content == "Question 3"

        with RequestProviderDataContext(user=user_r1):
            child_for_r1_after_delete = await store.get_response_object("resp-shared-child")
            assert len(child_for_r1_after_delete.input) == 5
            assert child_for_r1_after_delete.input[0].content == "Question 1"
            assert child_for_r1_after_delete.input[1].content[0].text == "Answer 1"
            assert child_for_r1_after_delete.input[2].content == "Question 2"
            assert child_for_r1_after_delete.input[3].content[0].text == "Answer 2"
            assert child_for_r1_after_delete.input[4].content == "Question 3"

        with RequestProviderDataContext(user=user_r2):
            child_for_r2_after_delete = await store.get_response_object("resp-shared-child")
            assert len(child_for_r2_after_delete.input) == 3
            assert child_for_r2_after_delete.input[0].content == "Question 2"
            assert child_for_r2_after_delete.input[1].content[0].text == "Answer 2"
            assert child_for_r2_after_delete.input[2].content == "Question 3"


async def test_incremental_upsert_preserves_materialized_snapshot_after_parent_delete():
    """Incremental upserts after parent deletion must not revert child rows back to delta-only input."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())
        parent = OpenAIResponseObject(
            id="resp-upsert-parent",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-upsert-parent",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            parent,
            [create_test_response_input("Question 1", "input-upsert-parent")],
            create_test_messages("Question 1"),
        )

        child = OpenAIResponseObject(
            id="resp-upsert-child",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[],
            status="in_progress",
            previous_response_id="resp-upsert-parent",
            store=True,
        )
        await store.store_response_object(
            child,
            [create_test_response_input("Question 2", "input-upsert-child")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        await store.delete_response_object("resp-upsert-parent")

        child_raw_after_delete = await store.get_response_object("resp-upsert-child", reconstruct_input=False)
        assert child_raw_after_delete.input_storage_mode is None
        assert child_raw_after_delete.previous_response_id is None
        assert len(child_raw_after_delete.input) == 3

        child_completed = OpenAIResponseObject(
            id="resp-upsert-child",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-upsert-child",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 2")],
                )
            ],
            status="completed",
            previous_response_id="resp-upsert-parent",
            store=True,
        )
        await store.upsert_response_object(
            child_completed,
            [create_test_response_input("Question 2", "input-upsert-child")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        child_raw_after_upsert = await store.get_response_object("resp-upsert-child", reconstruct_input=False)
        assert child_raw_after_upsert.input_storage_mode is None
        assert child_raw_after_upsert.previous_response_id is None
        assert len(child_raw_after_upsert.input) == 3
        assert child_raw_after_upsert.input[0].content == "Question 1"
        assert child_raw_after_upsert.input[1].content[0].text == "Answer 1"
        assert child_raw_after_upsert.input[2].content == "Question 2"

        child_after_upsert = await store.get_response_object("resp-upsert-child")
        assert len(child_after_upsert.input) == 3
        assert child_after_upsert.input[0].content == "Question 1"
        assert child_after_upsert.input[1].content[0].text == "Answer 1"
        assert child_after_upsert.input[2].content == "Question 2"


async def test_incremental_upsert_preserves_rewritten_ancestry_after_parent_delete():
    """Incremental upserts should not restore stale deleted parent links after rewrites."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())
        grandparent = OpenAIResponseObject(
            id="resp-rewire-grandparent",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-rewire-grandparent",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            grandparent,
            [create_test_response_input("Question 1", "input-rewire-grandparent")],
            create_test_messages("Question 1"),
        )

        parent = OpenAIResponseObject(
            id="resp-rewire-parent",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-rewire-parent",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 2")],
                )
            ],
            status="completed",
            previous_response_id="resp-rewire-grandparent",
            store=True,
        )
        await store.store_response_object(
            parent,
            [create_test_response_input("Question 2", "input-rewire-parent")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        child = OpenAIResponseObject(
            id="resp-rewire-child",
            created_at=base_time + 2,
            model="test-model",
            object="response",
            output=[],
            status="in_progress",
            previous_response_id="resp-rewire-parent",
            store=True,
        )
        await store.store_response_object(
            child,
            [create_test_response_input("Question 3", "input-rewire-child")],
            create_test_messages("Question 3"),
            incremental_input=True,
        )

        await store.delete_response_object("resp-rewire-parent")

        child_raw_after_delete = await store.get_response_object("resp-rewire-child", reconstruct_input=False)
        assert child_raw_after_delete.input_storage_mode == "incremental"
        assert child_raw_after_delete.previous_response_id == "resp-rewire-grandparent"
        assert len(child_raw_after_delete.input) == 3
        assert child_raw_after_delete.input[0].content == "Question 2"
        assert child_raw_after_delete.input[1].content[0].text == "Answer 2"
        assert child_raw_after_delete.input[2].content == "Question 3"

        child_completed = OpenAIResponseObject(
            id="resp-rewire-child",
            created_at=base_time + 2,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-rewire-child",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 3")],
                )
            ],
            status="completed",
            previous_response_id="resp-rewire-parent",
            store=True,
        )
        await store.upsert_response_object(
            child_completed,
            [create_test_response_input("Question 3", "input-rewire-child")],
            create_test_messages("Question 3"),
            incremental_input=True,
        )

        child_raw_after_upsert = await store.get_response_object("resp-rewire-child", reconstruct_input=False)
        assert child_raw_after_upsert.input_storage_mode == "incremental"
        assert child_raw_after_upsert.previous_response_id == "resp-rewire-grandparent"
        assert len(child_raw_after_upsert.input) == 3
        assert child_raw_after_upsert.input[0].content == "Question 2"
        assert child_raw_after_upsert.input[1].content[0].text == "Answer 2"
        assert child_raw_after_upsert.input[2].content == "Question 3"

        child_after_upsert = await store.get_response_object("resp-rewire-child")
        assert len(child_after_upsert.input) == 5
        assert child_after_upsert.input[0].content == "Question 1"
        assert child_after_upsert.input[1].content[0].text == "Answer 1"
        assert child_after_upsert.input[2].content == "Question 2"
        assert child_after_upsert.input[3].content[0].text == "Answer 2"
        assert child_after_upsert.input[4].content == "Question 3"


async def test_delete_response_denied_does_not_materialize_children():
    """Denied deletes should not mutate incremental child rows."""
    from ogx.core.access_control.access_control import AccessDeniedError
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        owner_policy = [AccessRule(permit=Scope(actions=[Action.CREATE, Action.READ]), when=["user is owner"])]
        store = build_store(db_path, policy=owner_policy)
        await store.initialize()

        user = User(principal="alice", attributes={"roles": ["member"]})
        base_time = int(time.time())

        with RequestProviderDataContext(user=user):
            parent = OpenAIResponseObject(
                id="resp-denied-parent",
                created_at=base_time,
                model="test-model",
                object="response",
                output=[
                    OpenAIResponseMessage(
                        id="out-denied-parent",
                        role="assistant",
                        content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                    )
                ],
                status="completed",
                store=True,
            )
            await store.store_response_object(
                parent,
                [create_test_response_input("Question 1", "input-denied-parent")],
                create_test_messages("Question 1"),
            )

            child = OpenAIResponseObject(
                id="resp-denied-child",
                created_at=base_time + 1,
                model="test-model",
                object="response",
                output=[],
                status="completed",
                previous_response_id="resp-denied-parent",
                store=True,
            )
            await store.store_response_object(
                child,
                [create_test_response_input("Question 2", "input-denied-child")],
                create_test_messages("Question 2"),
                incremental_input=True,
            )

            child_before_delete = await store.get_response_object("resp-denied-child", reconstruct_input=False)
            assert child_before_delete.input_storage_mode == "incremental"
            assert len(child_before_delete.input) == 1
            assert child_before_delete.input[0].content == "Question 2"

            with pytest.raises(AccessDeniedError):
                await store.delete_response_object("resp-denied-parent")

            child_after_denied_delete = await store.get_response_object("resp-denied-child", reconstruct_input=False)
            assert child_after_denied_delete.input_storage_mode == "incremental"
            assert len(child_after_denied_delete.input) == 1
            assert child_after_denied_delete.input[0].content == "Question 2"

            parent_after_denied_delete = await store.get_response_object("resp-denied-parent")
            assert parent_after_denied_delete.id == "resp-denied-parent"


async def test_list_responses_reuses_cached_ancestry_and_avoids_repeated_fetches():
    """list_responses should avoid repeated DB ancestry fetches for rows already in page."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        resp1 = OpenAIResponseObject(
            id="resp-cache-1",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-cache-1",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 1")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            resp1,
            [create_test_response_input("Question 1", "input-cache-1")],
            create_test_messages("Question 1"),
        )

        resp2 = OpenAIResponseObject(
            id="resp-cache-2",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-cache-2",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Answer 2")],
                )
            ],
            status="completed",
            previous_response_id="resp-cache-1",
            store=True,
        )
        await store.store_response_object(
            resp2,
            [create_test_response_input("Question 2", "input-cache-2")],
            create_test_messages("Question 2"),
            incremental_input=True,
        )

        resp3 = OpenAIResponseObject(
            id="resp-cache-3",
            created_at=base_time + 2,
            model="test-model",
            object="response",
            output=[],
            status="completed",
            previous_response_id="resp-cache-2",
            store=True,
        )
        await store.store_response_object(
            resp3,
            [create_test_response_input("Question 3", "input-cache-3")],
            create_test_messages("Question 3"),
            incremental_input=True,
        )

        await store.flush()

        original_fetch_one = store.sql_store.fetch_one
        store.sql_store.fetch_one = AsyncMock(side_effect=original_fetch_one)

        listed = await store.list_responses(limit=10, order=Order.desc)
        listed_resp3 = next(item for item in listed.data if item.id == "resp-cache-3")

        assert len(listed_resp3.input) == 5
        assert store.sql_store.fetch_one.await_count == 0


async def test_delete_response_queries_only_direct_children_for_materialization():
    """delete_response_object should query targeted direct children instead of full-table scans."""
    from ogx_api import OpenAIResponseMessage, OpenAIResponseOutputMessageContentOutputText

    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = build_store(db_path)
        await store.initialize()

        base_time = int(time.time())

        parent = OpenAIResponseObject(
            id="resp-target-parent",
            created_at=base_time,
            model="test-model",
            object="response",
            output=[
                OpenAIResponseMessage(
                    id="out-target-parent",
                    role="assistant",
                    content=[OpenAIResponseOutputMessageContentOutputText(text="Parent answer")],
                )
            ],
            status="completed",
            store=True,
        )
        await store.store_response_object(
            parent,
            [create_test_response_input("Parent question", "input-target-parent")],
            create_test_messages("Parent question"),
        )

        child = OpenAIResponseObject(
            id="resp-target-child",
            created_at=base_time + 1,
            model="test-model",
            object="response",
            output=[],
            status="completed",
            previous_response_id="resp-target-parent",
            store=True,
        )
        await store.store_response_object(
            child,
            [create_test_response_input("Child question", "input-target-child")],
            create_test_messages("Child question"),
            incremental_input=True,
        )

        await store.flush()

        original_fetch_all = store.sql_store.sql_store.fetch_all
        store.sql_store.sql_store.fetch_all = AsyncMock(side_effect=original_fetch_all)

        await store.delete_response_object("resp-target-parent")

        fetch_all_calls = [call.kwargs for call in store.sql_store.sql_store.fetch_all.await_args_list]
        materialization_calls = [
            kwargs
            for kwargs in fetch_all_calls
            if kwargs.get("where") == {"previous_response_id": "resp-target-parent"}
        ]
        assert len(materialization_calls) == 1
