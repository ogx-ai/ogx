# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ogx_api.schema_utils import json_schema_type
from ogx_api.shields import GetShieldRequest, Shield


@json_schema_type
class ModerationObjectResults(BaseModel):
    """A moderation result object containing flagged status and category information."""

    flagged: bool = Field(..., description="Whether any of the below categories are flagged")
    categories: dict[str, bool] | None = Field(
        None, description="A dictionary of the categories, and whether they are flagged or not"
    )
    category_applied_input_types: dict[str, list[str]] | None = Field(
        None, description="A dictionary of the categories along with the input type(s) that the score applies to"
    )
    category_scores: dict[str, float] | None = Field(
        None, description="A dictionary of the categories along with their scores as predicted by model"
    )
    user_message: str | None = Field(None, description="A message to convey to the user about the moderation result")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the moderation")


@json_schema_type
class ModerationObject(BaseModel):
    """A moderation object containing the results of content classification."""

    id: str = Field(..., description="The unique identifier for the moderation request")
    model: str = Field(..., description="The model used to generate the moderation results")
    results: list[ModerationObjectResults] = Field(..., description="A list of moderation result objects")


@json_schema_type
class ViolationLevel(Enum):
    """Severity level of a safety violation."""

    INFO = "info"  # Informational level violation that does not require action
    WARN = "warn"  # Warning level violation that suggests caution but allows continuation
    ERROR = "error"  # Error level violation that requires blocking or intervention


@json_schema_type
class SafetyViolation(BaseModel):
    """Details of a safety violation detected by content moderation."""

    violation_level: ViolationLevel = Field(..., description="Severity level of the violation")
    user_message: str | None = Field(None, description="Message to convey to the user about the violation")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata including specific violation codes"
    )


@json_schema_type
class RunShieldResponse(BaseModel):
    """Response from running a safety shield."""

    violation: SafetyViolation | None = Field(None, description="Safety violation detected by the shield, if any")


class ShieldStore(Protocol):
    """Protocol for accessing shields."""

    async def get_shield(self, request: GetShieldRequest) -> Shield: ...


__all__ = [
    "ModerationObjectResults",
    "ModerationObject",
    "ViolationLevel",
    "SafetyViolation",
    "RunShieldResponse",
    "ShieldStore",
]
