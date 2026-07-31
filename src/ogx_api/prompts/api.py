# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Prompts API protocol definition.

This module contains the Prompts protocol definition.
Pydantic models are defined in ogx_api.prompts.models.
The FastAPI router is defined in ogx_api.prompts.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from .models import (
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    ListPromptsResponse,
    ListPromptVersionsRequest,
    Prompt,
    SetDefaultVersionRequest,
    UpdatePromptRequest,
)


@runtime_checkable
class Prompts(Protocol):
    """Protocol for prompt management operations."""

    async def list_prompts(self) -> ListPromptsResponse: ...

    async def list_prompt_versions(self, request: ListPromptVersionsRequest) -> ListPromptsResponse: ...

    async def get_prompt(self, request: GetPromptRequest) -> Prompt: ...

    async def create_prompt(self, request: CreatePromptRequest) -> Prompt: ...

    async def update_prompt(self, request: UpdatePromptRequest) -> Prompt: ...

    async def delete_prompt(self, request: DeletePromptRequest) -> None: ...

    async def set_default_version(self, request: SetDefaultVersionRequest) -> Prompt: ...
