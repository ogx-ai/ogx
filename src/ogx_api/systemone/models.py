# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the SystemOne judgment/decision API.

These models follow the SystemOne wire format (POST /v1/systemone), the de facto standard for
serving judgment models that TypeSafe (the canonical implementation), Ollama, vLLM and
llama.cpp all converged on. No public OpenAPI document for this format exists, so these models
are kept permissive (``extra="allow"``) rather than strict: OGX forwards requests to and relays
responses from the configured backend largely as-is, the same way the Anthropic Messages API
passthrough does for a similarly undocumented-by-OGX wire format.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ogx_api.schema_utils import json_schema_type

# A question's `instructions` or the request's `state` may be a string, an object, or an array.
SystemOneText = str | dict[str, Any] | list[Any]


@json_schema_type
class SystemOneNoulQuestion(BaseModel):
    """A yes/no question, answered with a calibrated probability."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"] = "noul"
    instructions: SystemOneText = Field(..., description="The question to answer about `state`.")
    criteria: dict[str, str] | None = Field(default=None, description="Optional labels for the true/false outcomes.")


@json_schema_type
class SystemOneChoiceQuestion(BaseModel):
    """A question answered by picking one of up to 255 named options."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"] = "choice"
    instructions: SystemOneText = Field(..., description="The question to answer about `state`.")
    criteria: dict[str, str] = Field(..., description="Option name -> description, up to 255 options.")


@json_schema_type
class SystemOneScoreQuestion(BaseModel):
    """A question answered against an ordered rubric of 2-10 levels."""

    model_config = ConfigDict(extra="allow")

    type: Literal["score"] = "score"
    instructions: SystemOneText = Field(..., description="The question to answer about `state`.")
    criteria: dict[str, str] = Field(..., description="Ordered rubric level name -> description (2-10 levels).")


SystemOneQuestion = Annotated[
    SystemOneNoulQuestion | SystemOneChoiceQuestion | SystemOneScoreQuestion,
    Field(discriminator="type"),
]


@json_schema_type
class SystemOneRequest(BaseModel):
    """Request body for POST /v1/systemone."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The decision model to use for judgment.")
    state: SystemOneText = Field(..., description="The context the questions are evaluated against.")
    questions: dict[str, SystemOneQuestion] = Field(
        ..., description="Questions to answer about `state`, keyed by caller-chosen name."
    )


@json_schema_type
class SystemOneNoulAnswer(BaseModel):
    """Answer to a `noul` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0, le=1, description="Calibrated probability that the answer is true.")


@json_schema_type
class SystemOneChoiceAnswer(BaseModel):
    """Answer to a `choice` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="The selected option name, one of the question's `criteria` keys.")
    probabilities: dict[str, float] = Field(..., description="Probability assigned to each option.")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in `choice`.")


@json_schema_type
class SystemOneScoreAnswer(BaseModel):
    """Answer to a `score` question."""

    model_config = ConfigDict(extra="allow")

    type: Literal["score"] = "score"
    score: float = Field(..., description="Weighted score across the rubric's ordered levels.")
    legend: str | None = Field(default=None, description="Human-readable description of what `score` means.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="Probability assigned to each rubric level."
    )
    confidence: float | None = Field(default=None, ge=0, le=1, description="Overall confidence in `score`.")


SystemOneAnswer = Annotated[
    SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer,
    Field(discriminator="type"),
]


@json_schema_type
class SystemOneUsage(BaseModel):
    """Token usage for a SystemOne judgment call."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = Field(default=0, ge=0, description="Number of tokens in `state` and the questions.")
    output_tokens: int = Field(default=0, ge=0, description="Number of tokens used to produce the answers.")


@json_schema_type
class SystemOneResponse(BaseModel):
    """Response body for POST /v1/systemone."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The decision model that produced the answers.")
    answers: dict[str, SystemOneAnswer] = Field(
        ..., description="Answers to the request's questions, keyed by the same names."
    )
    usage: SystemOneUsage = Field(default_factory=SystemOneUsage, description="Token usage for this call.")
