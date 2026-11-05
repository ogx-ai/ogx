# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the SystemOne judgment/decision wire format (POST /v1/systemone).

These models follow the SystemOne wire format, the de facto standard for serving judgment
models that TypeSafe (the canonical implementation, whose API reference is the normative
spec), Ollama, vLLM and llama.cpp all converged on. Field types mirror the TypeSafe API
reference exactly, and the models are constrained (``extra="forbid"``): SystemOne is a closed
wire format with no extensions, so unknown fields are rejected rather than passed through to
the configured backend.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ogx_api.schema_utils import json_schema_type

# A question's `instructions` or the request's `state` may be a string, an object, or an array.
SystemOneText = str | dict[str, Any] | list[Any]


@json_schema_type
class SystemOneNoulCriteria(BaseModel):
    """Optional descriptions of the yes and no outcomes of a `noul` question."""

    model_config = ConfigDict(extra="forbid")

    true: SystemOneText | None = Field(default=None, description="What a yes (value near 1) means.")
    false: SystemOneText | None = Field(default=None, description="What a no (value near 0) means.")


@json_schema_type
class SystemOneNoulQuestion(BaseModel):
    """A yes/no question, answered with a calibrated probability."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: SystemOneText | None = Field(default=None, description="The question to answer about `state`.")
    criteria: SystemOneNoulCriteria | None = Field(
        default=None, description="Optional descriptions of the true/false outcomes."
    )


@json_schema_type
class SystemOneChoiceQuestion(BaseModel):
    """A question answered by picking one of up to 255 named options."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: SystemOneText | None = Field(default=None, description="The question to answer about `state`.")
    criteria: dict[str, SystemOneText | None] = Field(
        ...,
        max_length=255,
        description="Option name -> description (string, object, array, or null). Up to 255 options.",
    )


@json_schema_type
class SystemOneScoreQuestion(BaseModel):
    """A question answered against an ordered rubric of 1-10 levels."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: SystemOneText | None = Field(default=None, description="The question to answer about `state`.")
    criteria: list[SystemOneText] = Field(
        ..., min_length=1, max_length=10, description="Ordered list of level descriptions, one per score from zero."
    )


SystemOneQuestion = Annotated[
    SystemOneNoulQuestion | SystemOneChoiceQuestion | SystemOneScoreQuestion,
    Field(discriminator="type"),
]


@json_schema_type
class SystemOneRequest(BaseModel):
    """Request body for POST /v1/systemone."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="The decision model to use for judgment.")
    state: SystemOneText = Field(..., description="The context the questions are evaluated against.")
    questions: dict[str, SystemOneQuestion] = Field(
        ..., min_length=1, description="Questions to answer about `state`, keyed by caller-chosen name."
    )


@json_schema_type
class SystemOneNoulAnswer(BaseModel):
    """Answer to a `noul` question."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0, le=1, description="Calibrated probability that the answer is true.")


@json_schema_type
class SystemOneChoiceAnswer(BaseModel):
    """Answer to a `choice` question."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="The selected option name, one of the question's `criteria` keys.")
    probabilities: dict[str, float] = Field(..., description="Probability assigned to each option.")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in `choice`.")


@json_schema_type
class SystemOneScoreAnswer(BaseModel):
    """Answer to a `score` question."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    score: float = Field(
        ..., description="Probability-weighted score across the rubric's ordered levels; can land between levels."
    )
    legend: dict[str, SystemOneText] = Field(..., description="Each level (string key) mapped back to its description.")
    probabilities: dict[str, float] = Field(..., description="Probability assigned to each rubric level.")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in `score`.")


SystemOneAnswer = Annotated[
    SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer,
    Field(discriminator="type"),
]


@json_schema_type
class SystemOneUsage(BaseModel):
    """Token usage for a SystemOne judgment call."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0, description="Number of tokens in `state` and the questions.")
    output_tokens: int = Field(default=0, ge=0, description="Number of tokens used to produce the answers.")


@json_schema_type
class SystemOneResponse(BaseModel):
    """Response body for POST /v1/systemone."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="The decision model that produced the answers.")
    answers: dict[str, SystemOneAnswer] = Field(
        ..., description="Answers to the request's questions, keyed by the same names."
    )
    usage: SystemOneUsage = Field(default_factory=SystemOneUsage, description="Token usage for this call.")
