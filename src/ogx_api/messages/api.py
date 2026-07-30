# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .models import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
    CancelMessageBatchRequest,
    CreateMessageBatchRequest,
    ListMessageBatchesRequest,
    ListMessageBatchesResponse,
    MessageBatch,
    MessageBatchIndividualResponse,
    RetrieveMessageBatchRequest,
    RetrieveMessageBatchResultsRequest,
)


@runtime_checkable
class Messages(Protocol):
    """Protocol for the Anthropic Messages API."""

    async def create_message(
        self,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]: ...

    async def count_message_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse: ...

    async def create_message_batch(
        self,
        request: CreateMessageBatchRequest,
    ) -> MessageBatch: ...

    async def retrieve_message_batch(
        self,
        request: RetrieveMessageBatchRequest,
    ) -> MessageBatch: ...

    async def list_message_batches(
        self,
        request: ListMessageBatchesRequest,
    ) -> ListMessageBatchesResponse: ...

    async def cancel_message_batch(
        self,
        request: CancelMessageBatchRequest,
    ) -> MessageBatch: ...

    async def retrieve_message_batch_results(
        self,
        request: RetrieveMessageBatchResultsRequest,
    ) -> AsyncIterator[MessageBatchIndividualResponse]: ...
