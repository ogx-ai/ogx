# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Models API protocol definition.

This module contains the Models protocol definition.
Pydantic models are defined in ogx_api.models.models.
The FastAPI router is defined in ogx_api.models.fastapi_routes.
"""

from typing import Protocol, runtime_checkable

from .models import (
    AnthropicListModelsResponse,
    GetModelRequest,
    GoogleListModelsResponse,
    ListModelsResponse,
    Model,
    OpenAIListModelsResponse,
    RegisterModelRequest,
    UnregisterModelRequest,
)


@runtime_checkable
class Models(Protocol):
    """Protocol for model management operations."""

    async def list_models(self) -> ListModelsResponse: ...

    async def openai_list_models(self) -> OpenAIListModelsResponse: ...

    async def anthropic_list_models(self) -> AnthropicListModelsResponse: ...

    async def google_list_models(self) -> GoogleListModelsResponse: ...

    async def get_model(self, request: GetModelRequest) -> Model: ...

    async def register_model(self, request: RegisterModelRequest) -> Model: ...

    async def unregister_model(self, request: UnregisterModelRequest) -> None: ...
