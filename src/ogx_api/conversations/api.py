# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    AddItemsRequest,
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemDeletedResource,
    ConversationItemList,
    CreateConversationRequest,
    DeleteConversationRequest,
    DeleteItemRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
    UpdateConversationRequest,
)


@runtime_checkable
class Conversations(Protocol):
    """Protocol for conversation management operations."""

    async def create_conversation(self, request: CreateConversationRequest) -> Conversation: ...

    async def get_conversation(self, request: GetConversationRequest) -> Conversation: ...

    async def update_conversation(self, conversation_id: str, request: UpdateConversationRequest) -> Conversation: ...

    async def openai_delete_conversation(self, request: DeleteConversationRequest) -> ConversationDeletedResource: ...

    async def add_items(self, conversation_id: str, request: AddItemsRequest) -> ConversationItemList: ...

    async def retrieve(self, request: RetrieveItemRequest) -> ConversationItem: ...

    async def list_items(self, request: ListItemsRequest) -> ConversationItemList: ...

    async def openai_delete_conversation_item(self, request: DeleteItemRequest) -> ConversationItemDeletedResource: ...
