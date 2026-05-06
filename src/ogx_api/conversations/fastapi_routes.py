# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Conversations API.

This module defines the FastAPI router for the Conversations API using standard
FastAPI route decorators.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends, Path
from pydantic import BaseModel

from ogx_api.router_utils import (
    ExceptionTranslatingRoute,
    create_path_dependency,
    create_query_dependency,
    standard_responses,
)
from ogx_api.version import OGX_API_V1

from .api import Conversations
from .models import (
    AddItemsRequest,
    Conversation,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemDeletedResource,
    ConversationItemInclude,
    ConversationItemList,
    CreateConversationRequest,
    DeleteConversationRequest,
    DeleteItemRequest,
    GetConversationRequest,
    ListItemsRequest,
    RetrieveItemRequest,
    UpdateConversationRequest,
)


class _ListItemsQueryParams(BaseModel):
    """Query parameters for list_items endpoint (excludes conversation_id path param).

    This is a subset of ListItemsRequest that only includes query parameters,
    excluding the conversation_id which is a path parameter.
    """

    after: str | None = None
    include: list[ConversationItemInclude] | None = None
    limit: int | None = None
    order: Literal["asc", "desc"] | None = None


# Dependency functions for request models
get_conversation_request = create_path_dependency(GetConversationRequest)
delete_conversation_request = create_path_dependency(DeleteConversationRequest)
get_list_items_query_params = create_query_dependency(_ListItemsQueryParams)


def create_router(impl: Conversations) -> APIRouter:
    """Create a FastAPI router for the Conversations API."""
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Conversations"],
        responses=standard_responses,
        route_class=ExceptionTranslatingRoute,
    )

    @router.post(
        "/conversations",
        response_model=Conversation,
        summary="Create a conversation.",
        description="Create a conversation.",
        responses={200: {"description": "The created conversation object."}},
    )
    async def create_conversation(
        request: Annotated[CreateConversationRequest, Body(...)],
    ) -> Conversation:
        return await impl.create_conversation(request)

    @router.get(
        "/conversations/{conversation_id}",
        response_model=Conversation,
        summary="Retrieve a conversation.",
        description="Get a conversation with the given ID.",
        responses={200: {"description": "The conversation object."}},
    )
    async def get_conversation(
        request: Annotated[GetConversationRequest, Depends(get_conversation_request)],
    ) -> Conversation:
        return await impl.get_conversation(request)

    @router.post(
        "/conversations/{conversation_id}",
        response_model=Conversation,
        summary="Update a conversation.",
        description="Update a conversation's metadata with the given ID.",
        responses={200: {"description": "The updated conversation object."}},
    )
    async def update_conversation(
        conversation_id: Annotated[str, Path(description="The conversation identifier.")],
        request: Annotated[UpdateConversationRequest, Body(...)],
    ) -> Conversation:
        return await impl.update_conversation(conversation_id, request)

    @router.delete(
        "/conversations/{conversation_id}",
        response_model=ConversationDeletedResource,
        summary="Delete a conversation.",
        description="Delete a conversation with the given ID.",
        responses={200: {"description": "The deleted conversation resource."}},
    )
    async def delete_conversation(
        request: Annotated[DeleteConversationRequest, Depends(delete_conversation_request)],
    ) -> ConversationDeletedResource:
        return await impl.openai_delete_conversation(request)

    @router.post(
        "/conversations/{conversation_id}/items",
        response_model=ConversationItemList,
        summary="Create items.",
        description="Create items in the conversation.",
        responses={200: {"description": "List of created items."}},
    )
    async def add_items(
        conversation_id: Annotated[str, Path(description="The conversation identifier.")],
        request: Annotated[AddItemsRequest, Body(...)],
    ) -> ConversationItemList:
        return await impl.add_items(conversation_id, request)

    @router.get(
        "/conversations/{conversation_id}/items/{item_id}",
        response_model=ConversationItem,
        summary="Retrieve an item.",
        description="Retrieve a conversation item.",
        responses={200: {"description": "The conversation item."}},
    )
    async def retrieve_item(
        conversation_id: Annotated[str, Path(description="The conversation identifier.")],
        item_id: Annotated[str, Path(description="The item identifier.")],
    ) -> ConversationItem:
        request = RetrieveItemRequest(conversation_id=conversation_id, item_id=item_id)
        return await impl.retrieve(request)

    @router.get(
        "/conversations/{conversation_id}/items",
        response_model=ConversationItemList,
        summary="List items.",
        description="List items in the conversation.",
        responses={200: {"description": "List of conversation items."}},
    )
    async def list_items(
        conversation_id: Annotated[str, Path(description="The conversation identifier.")],
        query_params: Annotated[_ListItemsQueryParams, Depends(get_list_items_query_params)],
    ) -> ConversationItemList:
        request = ListItemsRequest(
            conversation_id=conversation_id,
            after=query_params.after,
            include=query_params.include,
            limit=query_params.limit,
            order=query_params.order,
        )
        return await impl.list_items(request)

    @router.delete(
        "/conversations/{conversation_id}/items/{item_id}",
        response_model=ConversationItemDeletedResource,
        summary="Delete an item.",
        description="Delete a conversation item.",
        responses={200: {"description": "The deleted item resource."}},
    )
    async def delete_item(
        conversation_id: Annotated[str, Path(description="The conversation identifier.")],
        item_id: Annotated[str, Path(description="The item identifier.")],
    ) -> ConversationItemDeletedResource:
        request = DeleteItemRequest(conversation_id=conversation_id, item_id=item_id)
        return await impl.openai_delete_conversation_item(request)

    return router
