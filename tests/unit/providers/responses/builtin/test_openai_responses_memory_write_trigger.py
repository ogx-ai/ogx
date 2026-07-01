# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from types import SimpleNamespace

from ogx.core.datatypes import User
from ogx.core.request_headers import RequestProviderDataContext
from ogx.providers.inline.responses.builtin.responses import openai_responses as openai_responses_module
from ogx_api import ConversationItemList
from ogx_api.responses.models import CreateResponseRequest, MemoryToolConfig
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_memory_write_trigger_schedules_for_completed_stored_conversation(
    monkeypatch,
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
    mock_conversations_api,
):
    recorded_calls = []

    async def fake_write_conversation_memory(**kwargs):
        recorded_calls.append(kwargs)

    def fake_create_detached_background_task(coro):
        return asyncio.create_task(coro)

    monkeypatch.setattr(openai_responses_module, "write_conversation_memory", fake_write_conversation_memory)
    monkeypatch.setattr(
        openai_responses_module,
        "create_detached_background_task",
        fake_create_detached_background_task,
    )
    conv_id = "conv_" + "c" * 48
    openai_responses_impl.memory_config.enabled = True
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"
    openai_responses_impl.memory_config.write_debounce_seconds = 0
    mock_conversations_api.list_items.return_value = ConversationItemList(
        data=[],
        first_id=None,
        has_more=False,
        last_id=None,
        object="list",
    )
    mock_responses_store.get_conversation_messages.return_value = None
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    response = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="remember this preference",
            model="test-model",
            conversation=conv_id,
            stream=False,
            memory=MemoryToolConfig(owner_id="user-123"),
        ),
    )
    await asyncio.sleep(0)

    assert response.status == "completed"
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["conversation_id"] == conv_id
    assert recorded_calls[0]["response_status"] == "completed"
    assert recorded_calls[0]["request_memory"] == MemoryToolConfig(owner_id="user-123")


async def test_memory_write_trigger_coalesces_pending_conversation_writes(
    monkeypatch,
    openai_responses_impl,
):
    recorded_calls = []

    async def fake_write_conversation_memory(**kwargs):
        recorded_calls.append(kwargs)

    def fake_create_detached_background_task(coro):
        return asyncio.create_task(coro)

    monkeypatch.setattr(openai_responses_module, "write_conversation_memory", fake_write_conversation_memory)
    monkeypatch.setattr(
        openai_responses_module,
        "create_detached_background_task",
        fake_create_detached_background_task,
    )
    openai_responses_impl.memory_config.enabled = True
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"
    openai_responses_impl.memory_config.write_debounce_seconds = 0.01

    openai_responses_impl._schedule_memory_write(
        conversation_id="conv_" + "g" * 48,
        response_id="resp_first",
        response_status="completed",
        model="test-model",
        memory=MemoryToolConfig(owner_id="user-123"),
        metadata=None,
        safety_identifier=None,
    )
    openai_responses_impl._schedule_memory_write(
        conversation_id="conv_" + "g" * 48,
        response_id="resp_second",
        response_status="completed",
        model="test-model",
        memory=MemoryToolConfig(owner_id="user-123"),
        metadata=None,
        safety_identifier=None,
    )
    await asyncio.sleep(0.05)

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["response_id"] == "resp_second"


async def test_memory_write_trigger_serializes_active_conversation_writes(
    monkeypatch,
    openai_responses_impl,
):
    calls = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def fake_write_conversation_memory(**kwargs):
        response_id = kwargs["response_id"]
        calls.append(("start", response_id))
        if response_id == "resp_first":
            first_started.set()
            await release_first.wait()
        calls.append(("end", response_id))

    def fake_create_detached_background_task(coro):
        return asyncio.create_task(coro)

    monkeypatch.setattr(openai_responses_module, "write_conversation_memory", fake_write_conversation_memory)
    monkeypatch.setattr(
        openai_responses_module,
        "create_detached_background_task",
        fake_create_detached_background_task,
    )
    openai_responses_impl.memory_config.enabled = True
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"
    openai_responses_impl.memory_config.write_debounce_seconds = 0

    openai_responses_impl._schedule_memory_write(
        conversation_id="conv_" + "i" * 48,
        response_id="resp_first",
        response_status="completed",
        model="test-model",
        memory=MemoryToolConfig(owner_id="user-123"),
        metadata=None,
        safety_identifier=None,
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    openai_responses_impl._schedule_memory_write(
        conversation_id="conv_" + "i" * 48,
        response_id="resp_second",
        response_status="completed",
        model="test-model",
        memory=MemoryToolConfig(owner_id="user-123"),
        metadata=None,
        safety_identifier=None,
    )
    await asyncio.sleep(0)

    assert calls == [("start", "resp_first")]

    release_first.set()
    for _ in range(10):
        if calls == [
            ("start", "resp_first"),
            ("end", "resp_first"),
            ("start", "resp_second"),
            ("end", "resp_second"),
        ]:
            break
        await asyncio.sleep(0.01)

    assert calls == [
        ("start", "resp_first"),
        ("end", "resp_first"),
        ("start", "resp_second"),
        ("end", "resp_second"),
    ]
    assert openai_responses_impl._memory_write_tasks == {}
    assert openai_responses_impl._memory_write_locks == {}


async def test_memory_write_trigger_captures_authenticated_owner_before_detaching(
    monkeypatch,
    openai_responses_impl,
):
    recorded_calls = []

    async def fake_write_conversation_memory(**kwargs):
        recorded_calls.append(kwargs)

    monkeypatch.setattr(openai_responses_module, "write_conversation_memory", fake_write_conversation_memory)
    openai_responses_impl.memory_config.enabled = True
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"
    openai_responses_impl.memory_config.write_debounce_seconds = 0

    with RequestProviderDataContext(user=User("auth-user", None)):
        openai_responses_impl._schedule_memory_write(
            conversation_id="conv_" + "h" * 48,
            response_id="resp_auth",
            response_status="completed",
            model="test-model",
            memory=MemoryToolConfig(),
            metadata=None,
            safety_identifier=None,
        )
    await asyncio.sleep(0)

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["owner_id"] == "auth-user"


async def test_memory_write_trigger_skips_when_store_false(
    monkeypatch,
    openai_responses_impl,
    mock_inference_api,
    mock_responses_store,
    mock_conversations_api,
):
    scheduled = []

    def fake_create_detached_background_task(coro):
        scheduled.append(coro)
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr(
        openai_responses_module,
        "create_detached_background_task",
        fake_create_detached_background_task,
    )
    conv_id = "conv_" + "d" * 48
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"
    mock_conversations_api.list_items.return_value = ConversationItemList(
        data=[],
        first_id=None,
        has_more=False,
        last_id=None,
        object="list",
    )
    mock_responses_store.get_conversation_messages.return_value = None
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    response = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="remember this preference",
            model="test-model",
            store=False,
            conversation=conv_id,
            stream=False,
            memory=MemoryToolConfig(owner_id="user-123"),
        ),
    )

    assert response.status == "completed"
    assert scheduled == []


def test_memory_write_trigger_skips_when_memory_config_disabled(
    monkeypatch,
    openai_responses_impl,
):
    scheduled = []

    def fake_create_detached_background_task(coro):
        scheduled.append(coro)
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr(
        openai_responses_module,
        "create_detached_background_task",
        fake_create_detached_background_task,
    )
    openai_responses_impl.memory_config.enabled = False
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"

    openai_responses_impl._schedule_memory_write(
        conversation_id="conv_" + "e" * 48,
        response_id="resp_123",
        response_status="completed",
        model="test-model",
        memory=MemoryToolConfig(owner_id="user-123"),
        metadata=None,
        safety_identifier=None,
    )

    assert scheduled == []


def test_memory_write_trigger_skips_when_request_memory_disabled(
    monkeypatch,
    openai_responses_impl,
):
    scheduled = []

    def fake_create_detached_background_task(coro):
        scheduled.append(coro)
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr(
        openai_responses_module,
        "create_detached_background_task",
        fake_create_detached_background_task,
    )
    openai_responses_impl.memory_config.enabled = True
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"

    openai_responses_impl._schedule_memory_write(
        conversation_id="conv_" + "f" * 48,
        response_id="resp_123",
        response_status="completed",
        model="test-model",
        memory=MemoryToolConfig(enabled=False, owner_id="user-123"),
        metadata=None,
        safety_identifier=None,
    )

    assert scheduled == []
