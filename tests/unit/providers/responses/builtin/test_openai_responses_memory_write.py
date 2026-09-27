# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import AsyncMock

from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx.providers.inline.responses.builtin.responses.memory import _format_memory_context, write_conversation_memory
from ogx_api import OpenAIFileObject, OpenAIUserMessageParam
from ogx_api.files.models import OpenAIFilePurpose
from ogx_api.responses.models import MemoryToolConfig
from ogx_api.vector_io.models import VectorStoreContent, VectorStoreSearchResponse


async def test_write_conversation_memory_skips_without_conversation():
    inference_api = AsyncMock()
    files_api = AsyncMock()
    vector_io_api = AsyncMock()
    responses_store = AsyncMock()

    await write_conversation_memory(
        inference_api=inference_api,
        files_api=files_api,
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=MemoryConfig(enabled=True, default_vector_store_id="vs_mem"),
        request_memory=MemoryToolConfig(owner_id="user-123"),
        conversation_id=None,
        response_id="resp_123",
        response_status="completed",
        model="test-model",
        metadata=None,
        safety_identifier=None,
    )

    inference_api.openai_chat_completion.assert_not_called()
    files_api.openai_upload_file.assert_not_called()
    vector_io_api.openai_attach_file_to_vector_store.assert_not_called()


async def test_write_conversation_memory_uploads_and_attaches_summary():
    inference_api = AsyncMock()
    files_api = AsyncMock()
    vector_io_api = AsyncMock()
    responses_store = AsyncMock()
    responses_store.get_conversation_messages.return_value = [OpenAIUserMessageParam(content="I prefer stacked PRs.")]
    responses_store.get_memory_record.return_value = None
    inference_api.openai_chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="User prefers stacked PRs."))]
    )
    files_api.openai_upload_file.return_value = OpenAIFileObject(
        id="file_new",
        bytes=42,
        created_at=123,
        filename="memory.md",
        purpose=OpenAIFilePurpose.ASSISTANTS,
        status="uploaded",
    )
    vector_io_api.openai_attach_file_to_vector_store.return_value = SimpleNamespace(status="completed")

    await write_conversation_memory(
        inference_api=inference_api,
        files_api=files_api,
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=MemoryConfig(enabled=True, default_vector_store_id="vs_mem"),
        request_memory=MemoryToolConfig(owner_id="user-123"),
        conversation_id="conv_abc",
        response_id="resp_123",
        response_status="completed",
        model="test-model",
        metadata=None,
        safety_identifier=None,
    )

    upload_file = files_api.openai_upload_file.call_args.kwargs["file"]
    uploaded_content = upload_file.file.getvalue().decode("utf-8")
    assert "User prefers stacked PRs." in uploaded_content

    attach_request = vector_io_api.openai_attach_file_to_vector_store.call_args.kwargs["request"]
    assert attach_request.file_id == "file_new"
    assert attach_request.attributes["memory"] is True
    assert attach_request.attributes["owner_id"] == "user-123"
    assert attach_request.attributes["conversation_id"] == "conv_abc"
    assert attach_request.attributes["response_id"] == "resp_123"
    responses_store.upsert_memory_record.assert_awaited_once_with(
        owner_id="user-123",
        conversation_id="conv_abc",
        vector_store_id="vs_mem",
        file_id="file_new",
        response_id="resp_123",
    )


async def test_write_conversation_memory_bounds_summary_input_and_uploaded_transcript():
    inference_api = AsyncMock()
    files_api = AsyncMock()
    vector_io_api = AsyncMock()
    responses_store = AsyncMock()
    responses_store.get_conversation_messages.return_value = [
        OpenAIUserMessageParam(content=f"turn {idx} " + ("x" * 100)) for idx in range(5)
    ]
    responses_store.get_memory_record.return_value = None
    inference_api.openai_chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Recent bounded summary."))]
    )
    files_api.openai_upload_file.return_value = OpenAIFileObject(
        id="file_new",
        bytes=42,
        created_at=123,
        filename="memory.md",
        purpose=OpenAIFilePurpose.ASSISTANTS,
        status="uploaded",
    )
    vector_io_api.openai_attach_file_to_vector_store.return_value = SimpleNamespace(status="completed")

    await write_conversation_memory(
        inference_api=inference_api,
        files_api=files_api,
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=MemoryConfig(
            enabled=True,
            default_vector_store_id="vs_mem",
            max_summary_messages=2,
            max_transcript_chars=80,
        ),
        request_memory=MemoryToolConfig(owner_id="user-123"),
        conversation_id="conv_abc",
        response_id="resp_123",
        response_status="completed",
        model="test-model",
        metadata=None,
        safety_identifier=None,
    )

    summary_request = inference_api.openai_chat_completion.await_args.args[0]
    assert [message.content for message in summary_request.messages[:-1]] == [
        "turn 3 " + ("x" * 100),
        "turn 4 " + ("x" * 100),
    ]
    upload_file = files_api.openai_upload_file.call_args.kwargs["file"]
    uploaded_content = upload_file.file.getvalue().decode("utf-8")
    assert "turn 0" not in uploaded_content
    assert "turn 3" in uploaded_content
    assert "[truncated]" in uploaded_content


async def test_write_conversation_memory_deletes_previous_memory_file_object():
    inference_api = AsyncMock()
    files_api = AsyncMock()
    vector_io_api = AsyncMock()
    responses_store = AsyncMock()
    responses_store.get_conversation_messages.return_value = [OpenAIUserMessageParam(content="I prefer stacked PRs.")]
    responses_store.get_memory_record.return_value = SimpleNamespace(file_id="file_old")
    inference_api.openai_chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="User prefers stacked PRs."))]
    )
    files_api.openai_upload_file.return_value = OpenAIFileObject(
        id="file_new",
        bytes=42,
        created_at=123,
        filename="memory.md",
        purpose=OpenAIFilePurpose.ASSISTANTS,
        status="uploaded",
    )
    vector_io_api.openai_attach_file_to_vector_store.return_value = SimpleNamespace(status="completed")

    await write_conversation_memory(
        inference_api=inference_api,
        files_api=files_api,
        vector_io_api=vector_io_api,
        responses_store=responses_store,
        memory_config=MemoryConfig(enabled=True, default_vector_store_id="vs_mem"),
        request_memory=MemoryToolConfig(owner_id="user-123"),
        conversation_id="conv_abc",
        response_id="resp_123",
        response_status="completed",
        model="test-model",
        metadata=None,
        safety_identifier=None,
    )

    vector_io_api.openai_delete_vector_store_file.assert_awaited_once_with(
        vector_store_id="vs_mem",
        file_id="file_old",
    )
    delete_request = files_api.openai_delete_file.await_args.kwargs["request"]
    assert delete_request.file_id == "file_old"


def test_format_memory_context_skips_when_budget_cannot_fit_framing():
    context = _format_memory_context(
        memory_config=MemoryConfig(),
        results=[
            VectorStoreSearchResponse(
                file_id="file_1",
                filename="memory.md",
                score=0.9,
                attributes={"created_at": 123},
                content=[VectorStoreContent(type="text", text="Useful memory.")],
            )
        ],
        max_context_tokens=1,
    )

    assert context is None
