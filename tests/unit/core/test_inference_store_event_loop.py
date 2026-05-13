# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests that InferenceStore works across event loop boundaries after engine reset.

Simulates the actual server startup pattern where Stack.initialize() runs in a
temporary event loop and request handling runs in uvicorn's event loop.
"""

import asyncio

from ogx.core.storage.datatypes import InferenceStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends, reset_sqlstore_engines
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIUserMessageParam,
)


def _make_completion(completion_id: str, created: int = 1000) -> OpenAIChatCompletion:
    return OpenAIChatCompletion(
        id=completion_id,
        created=created,
        model="test-model",
        object="chat.completion",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIChatCompletionResponseMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
    )


def _make_messages():
    return [OpenAIUserMessageParam(role="user", content="hi")]


def test_inference_store_data_persisted_after_event_loop_reset(tmp_path):
    """Data written via store_chat_completion is retrievable after engine reset.

    Uses direct writes (SQLite disables the write queue) to verify that the
    SQL engine is properly recreated and data reaches the database.
    """
    db_path = str(tmp_path / "inference.db")
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=db_path)})

    reference = InferenceStoreReference(backend="sql_default", table_name="inference_store")

    async def init_phase():
        store = InferenceStore(reference, policy=[])
        await store.initialize()
        return store

    store = asyncio.run(init_phase())

    reset_sqlstore_engines()

    async def request_phase():
        await store.store_chat_completion(_make_completion("cmpl-1"), _make_messages())
        await store.flush()
        result = await store.list_chat_completions()
        return result

    result = asyncio.run(request_phase())
    assert len(result.data) == 1
    assert result.data[0].id == "cmpl-1"
    assert result.data[0].choices[0].message.content == "hello"
    assert len(result.data[0].input_messages) == 1


def test_inference_store_write_queue_after_event_loop_reset(tmp_path):
    """Write queue workers function correctly after engine reset.

    Forces enable_write_queue=True (normally disabled for SQLite) to exercise
    the background worker path that fails with 'Future attached to a different
    loop' when the engine is bound to the wrong event loop.
    """
    db_path = str(tmp_path / "inference.db")
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=db_path)})

    reference = InferenceStoreReference(backend="sql_default", table_name="inference_store")

    async def init_phase():
        store = InferenceStore(reference, policy=[])
        await store.initialize()
        return store

    store = asyncio.run(init_phase())

    reset_sqlstore_engines()

    async def request_phase():
        # Force write queue on with a single writer to simulate PostgreSQL
        # behavior. Multiple writers race on SQLite, so use one worker.
        store.enable_write_queue = True
        store._num_writers = 1
        store._queue = None
        store._worker_tasks = []

        for i in range(3):
            await store.store_chat_completion(_make_completion(f"cmpl-{i}", created=1000 + i), _make_messages())
        await store.flush()

        result = await store.list_chat_completions()
        await store.shutdown()
        return result

    result = asyncio.run(request_phase())
    assert len(result.data) == 3
    ids = {r.id for r in result.data}
    assert ids == {"cmpl-0", "cmpl-1", "cmpl-2"}


def test_inference_store_retrieve_after_event_loop_reset(tmp_path):
    """Individual completion retrieval works after engine reset."""
    db_path = str(tmp_path / "inference.db")
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=db_path)})

    reference = InferenceStoreReference(backend="sql_default", table_name="inference_store")

    async def init_phase():
        store = InferenceStore(reference, policy=[])
        await store.initialize()
        return store

    store = asyncio.run(init_phase())

    reset_sqlstore_engines()

    async def request_phase():
        await store.store_chat_completion(_make_completion("cmpl-42"), _make_messages())
        await store.flush()

        completion = await store.get_chat_completion("cmpl-42")
        return completion

    completion = asyncio.run(request_phase())
    assert completion.id == "cmpl-42"
    assert completion.choices[0].message.content == "hello"
    assert completion.input_messages[0].content == "hi"
