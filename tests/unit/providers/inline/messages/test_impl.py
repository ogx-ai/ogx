# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for BuiltinMessagesImpl — batch operations and inference delegation."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from ogx.core.storage.datatypes import KVStoreReference
from ogx.providers.inline.messages.config import MessagesConfig
from ogx.providers.inline.messages.impl import BuiltinMessagesImpl
from ogx_api.messages.models import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessage,
    AnthropicMessageResponse,
    AnthropicUsage,
    CancelMessageBatchRequest,
    CreateMessageBatchRequest,
    MessageBatch,
    MessageBatchRequestParams,
    RetrieveMessageBatchResultsRequest,
)


@pytest.fixture
def impl():
    mock_inference = AsyncMock()
    mock_kvstore = AsyncMock()
    config = MessagesConfig(kvstore=KVStoreReference(backend="kv_default", namespace="test"))
    return BuiltinMessagesImpl(config=config, inference_api=mock_inference, kvstore=mock_kvstore)


class TestCreateMessageDelegation:
    async def test_create_message_delegates_to_inference_api(self, impl):
        request = AnthropicCreateMessageRequest(
            model="claude-sonnet-4-20250514",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
        )
        expected_response = AnthropicMessageResponse(
            id="msg_abc",
            content=[],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )
        impl.inference_api.anthropic_messages.return_value = expected_response

        result = await impl.create_message(request)

        impl.inference_api.anthropic_messages.assert_awaited_once_with(request)
        assert result is expected_response


class TestCountTokensDelegation:
    async def test_count_message_tokens_delegates_to_inference_api(self, impl):
        request = AnthropicCountTokensRequest(
            model="claude-sonnet-4-20250514",
            system="You are helpful.",
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        expected_response = AnthropicCountTokensResponse(input_tokens=10)
        impl.inference_api.anthropic_count_tokens.return_value = expected_response

        result = await impl.count_message_tokens(request)

        impl.inference_api.anthropic_count_tokens.assert_awaited_once_with(request)
        assert result is expected_response


class TestMessageBatchCreation:
    async def test_create_batch_detects_duplicate_custom_ids(self, impl):
        request = CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id="dup-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
                MessageBatchRequestParams(
                    custom_id="dup-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ],
        )

        with pytest.raises(ValueError, match="duplicate custom_id"):
            await impl.create_message_batch(request)

    async def test_create_batch_returns_in_progress_status(self, impl):
        request = CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id="req-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ],
        )

        batch = await impl.create_message_batch(request)

        assert batch.id.startswith("msgbatch_")
        assert batch.processing_status == "in_progress"
        assert batch.request_counts.processing == 1

    async def test_cancel_already_ended_batch_raises(self, impl):
        request = CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id="req-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ],
        )

        batch = await impl.create_message_batch(request)
        batch_id = batch.id

        from datetime import UTC, datetime

        ended_batch = MessageBatch(
            id=batch_id,
            processing_status="ended",
            request_counts=batch.request_counts,
            created_at=batch.created_at,
            expires_at=batch.expires_at,
            ended_at=datetime.now(UTC).isoformat(),
        )
        impl.kvstore.get = AsyncMock(return_value=ended_batch.model_dump_json())

        with pytest.raises(ValueError, match="batch has already ended"):
            await impl.cancel_message_batch(CancelMessageBatchRequest(batch_id=batch_id))

    async def test_retrieve_results_not_ended_raises(self, impl):
        request = CreateMessageBatchRequest(
            requests=[
                MessageBatchRequestParams(
                    custom_id="req-1",
                    params=AnthropicCreateMessageRequest(
                        model="m",
                        messages=[AnthropicMessage(role="user", content="Hi")],
                        max_tokens=100,
                    ),
                ),
            ],
        )

        batch = await impl.create_message_batch(request)
        batch_id = batch.id

        # Cancel the background processing task so it doesn't interfere with mocks
        task = impl._processing_tasks.pop(batch_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Mock kvstore to return an in-progress batch
        impl.kvstore.get = AsyncMock(
            return_value=MessageBatch(
                id=batch_id,
                processing_status="in_progress",
                request_counts=batch.request_counts,
                created_at=batch.created_at,
                expires_at=batch.expires_at,
            ).model_dump_json()
        )

        with pytest.raises(ValueError, match="batch has not finished processing"):
            results_iter = await impl.retrieve_message_batch_results(
                RetrieveMessageBatchResultsRequest(batch_id=batch_id)
            )
            await results_iter.__anext__()
