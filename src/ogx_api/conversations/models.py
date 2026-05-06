# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Conversations API requests and responses.

This module defines the request and response models for the Conversations API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from ogx_api.openai_responses import (
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageReasoningItem,
    OpenAIResponseOutputMessageWebSearchToolCall,
)
from ogx_api.schema_utils import json_schema_type, register_schema

Metadata = dict[str, str]


@json_schema_type
class Conversation(BaseModel):
    """OpenAI-compatible conversation object."""

    id: str = Field(..., description="The unique ID of the conversation.")
    object: Literal["conversation"] = Field(
        default="conversation", description="The object type, which is always conversation."
    )
    created_at: int = Field(
        ..., description="The time at which the conversation was created, measured in seconds since the Unix epoch."
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.",
    )
    items: list[dict] | None = Field(
        default=None,
        description="Initial items to include in the conversation context. You may add up to 20 items at a time.",
    )


@json_schema_type
class ConversationMessage(BaseModel):
    """OpenAI-compatible message item for conversations."""

    id: str = Field(..., description="unique identifier for this message")
    content: list[dict] = Field(..., description="message content")
    role: str = Field(..., description="message role")
    status: str = Field(..., description="message status")
    type: Literal["message"] = "message"
    object: Literal["message"] = "message"


ConversationItem = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseInputFunctionToolCallOutput
    | OpenAIResponseMCPApprovalRequest
    | OpenAIResponseMCPApprovalResponse
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools
    | OpenAIResponseOutputMessageReasoningItem,
    Field(discriminator="type"),
]
register_schema(ConversationItem, name="ConversationItem")


def _ensure_item_type(item: Any) -> Any:
    if isinstance(item, dict) and "type" not in item and "role" in item:
        return {**item, "type": "message"}
    return item


@json_schema_type
class ConversationDeletedResource(BaseModel):
    """Response for deleted conversation."""

    id: str = Field(..., description="The deleted conversation identifier")
    object: str = Field(default="conversation.deleted", description="Object type")
    deleted: bool = Field(default=True, description="Whether the object was deleted")


@json_schema_type
class ConversationItemCreateRequest(BaseModel):
    """Request body for creating conversation items."""

    items: list[ConversationItem] = Field(
        ...,
        description="Items to include in the conversation context. You may add up to 20 items at a time.",
        max_length=20,
    )

    @model_validator(mode="before")
    @classmethod
    def default_message_type(cls, value: Any) -> Any:
        if isinstance(value, dict) and "items" in value:
            value["items"] = [_ensure_item_type(item) for item in value["items"]]
        return value


class ConversationItemInclude(StrEnum):
    """Specify additional output data to include in the model response."""

    web_search_call_action_sources = "web_search_call.action.sources"
    code_interpreter_call_outputs = "code_interpreter_call.outputs"
    computer_call_output_output_image_url = "computer_call_output.output.image_url"
    file_search_call_results = "file_search_call.results"
    message_input_image_image_url = "message.input_image.image_url"
    message_output_text_logprobs = "message.output_text.logprobs"
    reasoning_encrypted_content = "reasoning.encrypted_content"


@json_schema_type
class ConversationItemList(BaseModel):
    """List of conversation items with pagination."""

    object: str = Field(default="list", description="Object type")
    data: list[ConversationItem] = Field(..., description="List of conversation items")
    first_id: str | None = Field(default=None, description="The ID of the first item in the list")
    last_id: str | None = Field(default=None, description="The ID of the last item in the list")
    has_more: bool = Field(default=False, description="Whether there are more items available")


@json_schema_type
class ConversationItemDeletedResource(BaseModel):
    """Response for deleted conversation item."""

    id: str = Field(..., description="The deleted item identifier")
    object: str = Field(default="conversation.item.deleted", description="Object type")
    deleted: bool = Field(default=True, description="Whether the object was deleted")


# Request models for each endpoint


@json_schema_type
class CreateConversationRequest(BaseModel):
    """Request model for creating a conversation."""

    items: list[ConversationItem] | None = Field(
        default=None,
        description="Initial items to include in the conversation context.",
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Set of key-value pairs that can be attached to an object.",
    )

    @model_validator(mode="before")
    @classmethod
    def default_message_type(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("items") is not None:
            value["items"] = [_ensure_item_type(item) for item in value["items"]]
        return value


@json_schema_type
class GetConversationRequest(BaseModel):
    """Request model for getting a conversation by ID."""

    conversation_id: str = Field(..., description="The conversation identifier.")


@json_schema_type
class UpdateConversationRequest(BaseModel):
    """Request model for updating a conversation's metadata."""

    metadata: Metadata = Field(
        ...,
        description="Set of key-value pairs that can be attached to an object.",
    )


@json_schema_type
class DeleteConversationRequest(BaseModel):
    """Request model for deleting a conversation."""

    conversation_id: str = Field(..., description="The conversation identifier.")


@json_schema_type
class AddItemsRequest(BaseModel):
    """Request model for adding items to a conversation."""

    items: list[ConversationItem] = Field(
        ...,
        description="Items to include in the conversation context. You may add up to 20 items at a time.",
        max_length=20,
    )

    @model_validator(mode="before")
    @classmethod
    def default_message_type(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value["items"] = [_ensure_item_type(item) for item in value.get("items", [])]
        return value


@json_schema_type
class RetrieveItemRequest(BaseModel):
    """Request model for retrieving a conversation item."""

    conversation_id: str = Field(..., description="The conversation identifier.")
    item_id: str = Field(..., description="The item identifier.")


@json_schema_type
class ListItemsRequest(BaseModel):
    """Request model for listing items in a conversation."""

    conversation_id: str = Field(..., description="The conversation identifier.")
    after: str | None = Field(
        default=None,
        description="An item ID to list items after, used in pagination.",
    )
    include: list[ConversationItemInclude] | None = Field(
        default=None,
        description="Specify additional output data to include in the response.",
    )
    limit: int | None = Field(
        default=None,
        description="A limit on the number of objects to be returned (1-100, default 20).",
    )
    order: Literal["asc", "desc"] | None = Field(
        default=None,
        description="The order to return items in (asc or desc, default desc).",
    )


@json_schema_type
class DeleteItemRequest(BaseModel):
    """Request model for deleting a conversation item."""

    conversation_id: str = Field(..., description="The conversation identifier.")
    item_id: str = Field(..., description="The item identifier.")


__all__ = [
    "Metadata",
    "Conversation",
    "ConversationMessage",
    "ConversationItem",
    "ConversationDeletedResource",
    "ConversationItemCreateRequest",
    "ConversationItemInclude",
    "ConversationItemList",
    "ConversationItemDeletedResource",
    "CreateConversationRequest",
    "GetConversationRequest",
    "UpdateConversationRequest",
    "DeleteConversationRequest",
    "AddItemsRequest",
    "RetrieveItemRequest",
    "ListItemsRequest",
    "DeleteItemRequest",
]
