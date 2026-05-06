# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Prompts API.

This module defines the FastAPI router for the Prompts API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Query

from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Prompts
from .models import (
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    ListPromptsResponse,
    ListPromptVersionsRequest,
    Prompt,
    SetDefaultVersionBodyRequest,
    SetDefaultVersionRequest,
    UpdatePromptBodyRequest,
    UpdatePromptRequest,
)

# Path parameter dependencies for single-field models
list_prompt_versions_request = create_path_dependency(ListPromptVersionsRequest)
delete_prompt_request = create_path_dependency(DeletePromptRequest)


def create_router(impl: Prompts) -> APIRouter:
    """Create a FastAPI router for the Prompts API.

    Args:
        impl: The Prompts implementation instance

    Returns:
        APIRouter configured for the Prompts API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Prompts"],
        responses=standard_responses,
    )

    @router.get(
        "/prompts",
        response_model=ListPromptsResponse,
        summary="List all prompts.",
        description="List all prompts.",
        responses={
            200: {"description": "A ListPromptsResponse containing all prompts."},
        },
    )
    async def list_prompts() -> ListPromptsResponse:
        return await impl.list_prompts()

    @router.get(
        "/prompts/{prompt_id}/versions",
        response_model=ListPromptsResponse,
        summary="List prompt versions.",
        description="List all versions of a specific prompt.",
        responses={
            200: {"description": "A ListPromptsResponse containing all versions of the prompt."},
        },
    )
    async def list_prompt_versions(
        request: Annotated[ListPromptVersionsRequest, Depends(list_prompt_versions_request)],
    ) -> ListPromptsResponse:
        return await impl.list_prompt_versions(request)

    @router.get(
        "/prompts/{prompt_id}",
        response_model=Prompt,
        summary="Get a prompt.",
        description="Get a prompt by its identifier and optional version.",
        responses={
            200: {"description": "A Prompt resource."},
        },
    )
    async def get_prompt(
        prompt_id: Annotated[str, Path(description="The identifier of the prompt to get.")],
        version: Annotated[
            int | None, Query(description="The version of the prompt to get (defaults to latest).")
        ] = None,
    ) -> Prompt:
        request = GetPromptRequest(prompt_id=prompt_id, version=version)
        return await impl.get_prompt(request)

    @router.post(
        "/prompts",
        response_model=Prompt,
        summary="Create a prompt.",
        description="Create a new prompt.",
        responses={
            200: {"description": "The created Prompt resource."},
        },
    )
    async def create_prompt(
        request: Annotated[CreatePromptRequest, Body(...)],
    ) -> Prompt:
        return await impl.create_prompt(request)

    @router.put(
        "/prompts/{prompt_id}",
        response_model=Prompt,
        summary="Update a prompt.",
        description="Update an existing prompt (increments version).",
        responses={
            200: {"description": "The updated Prompt resource with incremented version."},
        },
    )
    async def update_prompt(
        prompt_id: Annotated[str, Path(description="The identifier of the prompt to update.")],
        body: Annotated[UpdatePromptBodyRequest, Body(...)],
    ) -> Prompt:
        request = UpdatePromptRequest(
            prompt_id=prompt_id,
            prompt=body.prompt,
            version=body.version,
            variables=body.variables,
            set_as_default=body.set_as_default,
        )
        return await impl.update_prompt(request)

    @router.delete(
        "/prompts/{prompt_id}",
        summary="Delete a prompt.",
        description="Delete a prompt.",
        responses={
            200: {"description": "The prompt was successfully deleted."},
        },
    )
    async def delete_prompt(
        request: Annotated[DeletePromptRequest, Depends(delete_prompt_request)],
    ) -> None:
        return await impl.delete_prompt(request)

    @router.put(
        "/prompts/{prompt_id}/set-default-version",
        response_model=Prompt,
        summary="Set prompt version.",
        description="Set which version of a prompt should be the default in get_prompt (latest).",
        responses={
            200: {"description": "The prompt with the specified version now set as default."},
        },
    )
    async def set_default_version(
        prompt_id: Annotated[str, Path(description="The identifier of the prompt.")],
        body: Annotated[SetDefaultVersionBodyRequest, Body(...)],
    ) -> Prompt:
        request = SetDefaultVersionRequest(prompt_id=prompt_id, version=body.version)
        return await impl.set_default_version(request)

    return router
