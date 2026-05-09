# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Messages API protocol and models.

This module contains the Messages protocol definition for the Anthropic Messages API.
Pydantic models are defined in ogx_api.messages.models.
The FastAPI router is defined in ogx_api.messages.fastapi_routes.
"""

from . import fastapi_routes
from .api import Messages
from .models import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicErrorResponse,
    AnthropicImageBlock,
    AnthropicImageSource,
    AnthropicMessage,
    AnthropicMessageResponse,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicThinkingConfig,
    AnthropicToolDef,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
)

__all__ = [
    "Messages",
    "AnthropicContentBlock",
    "AnthropicCountTokensRequest",
    "AnthropicCountTokensResponse",
    "AnthropicCreateMessageRequest",
    "AnthropicErrorResponse",
    "AnthropicImageBlock",
    "AnthropicImageSource",
    "AnthropicMessage",
    "AnthropicMessageResponse",
    "AnthropicTextBlock",
    "AnthropicThinkingBlock",
    "AnthropicThinkingConfig",
    "AnthropicToolDef",
    "AnthropicToolResultBlock",
    "AnthropicToolUseBlock",
    "AnthropicUsage",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "MessageDeltaEvent",
    "MessageStartEvent",
    "MessageStopEvent",
    "fastapi_routes",
]
