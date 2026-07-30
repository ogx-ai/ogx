# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Conversations API protocol and models.

This module contains the Conversations protocol definition.
Pydantic models are defined in ogx_api.conversations.models.
The FastAPI router is defined in ogx_api.conversations.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Conversations

# Import models for re-export
from .models import (
    AddItemsRequest,
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemCreateRequest,
    ConversationItemDeletedResource,
    ConversationItemInclude,
    ConversationItemList,
    ConversationMessage,
    CreateConversationRequest,
    DeleteConversationRequest,
    DeleteItemRequest,
    GetConversationRequest,
    ListItemsRequest,
    Metadata,
    RetrieveItemRequest,
    UpdateConversationRequest,
)

__all__ = [
    "Conversations",
    "Conversation",
    "ConversationMessage",
    "ConversationItem",
    "ConversationDeletedResource",
    "ConversationItemCreateRequest",
    "ConversationItemInclude",
    "ConversationItemList",
    "ConversationItemDeletedResource",
    "Metadata",
    "CreateConversationRequest",
    "GetConversationRequest",
    "UpdateConversationRequest",
    "DeleteConversationRequest",
    "AddItemsRequest",
    "RetrieveItemRequest",
    "ListItemsRequest",
    "DeleteItemRequest",
    "fastapi_routes",
]
