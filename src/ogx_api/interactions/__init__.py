# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Interactions API protocol and models.

This module contains the Interactions protocol definition for the Google Interactions API.
Pydantic models are defined in ogx_api.interactions.models.
The FastAPI router is defined in ogx_api.interactions.fastapi_routes.
"""

from . import fastapi_routes
from .api import Interactions
from .models import (
    ContentDeltaEvent,
    ContentStartEvent,
    ContentStopEvent,
    GoogleCreateInteractionRequest,
    GoogleErrorResponse,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleInteractionResponse,
    GoogleStreamEvent,
    GoogleTextContent,
    GoogleTextOutput,
    GoogleUsage,
    InteractionCompleteEvent,
    InteractionStartEvent,
)

__all__ = [
    "Interactions",
    "ContentDeltaEvent",
    "ContentStartEvent",
    "ContentStopEvent",
    "GoogleCreateInteractionRequest",
    "GoogleErrorResponse",
    "GoogleGenerationConfig",
    "GoogleInputTurn",
    "GoogleInteractionResponse",
    "GoogleStreamEvent",
    "GoogleTextContent",
    "GoogleTextOutput",
    "GoogleUsage",
    "InteractionCompleteEvent",
    "InteractionStartEvent",
    "fastapi_routes",
]
