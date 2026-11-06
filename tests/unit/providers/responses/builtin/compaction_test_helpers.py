# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared helpers for the auto-compaction test suite (test_auto_compaction.py and
test_auto_compaction_followups.py).
"""

from unittest.mock import AsyncMock

from ogx.providers.utils.responses.responses_store import _OpenAIResponseObjectWithInputAndMessages
from ogx_api.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputMessageContentText,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


def _compaction_completion(
    summary: str | None, prompt_tokens: int = 50, completion_tokens: int = 20, finish_reason: str = "stop"
):
    """A non-streaming OpenAIChatCompletion, matching the shape _build_compaction's
    summarization call (stream=False) expects back."""
    return OpenAIChatCompletion(
        id="chatcmpl-compact",
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason=finish_reason,
                message=OpenAIChatCompletionResponseMessage(role="assistant", content=summary),
            )
        ],
        created=0,
        model="test-model",
        usage=OpenAIChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _dual_inference_mock(summary: str | None = "Summary of prior turns.", **usage_kwargs):
    """Route stream=False calls (the summarization call) to a canned completion, and
    stream=True calls (the main turn) to fake_stream() -- the two calls a compacting
    turn makes are otherwise indistinguishable to a single mocked return_value.
    """

    async def side_effect(params):
        if params.stream:
            return fake_stream()
        return _compaction_completion(summary, **usage_kwargs)

    return AsyncMock(side_effect=side_effect)


def _previous_response(
    *,
    input_text: str = "What is 2+2?",
    output_text: str = "2+2 equals 4.",
    usage: OpenAIChatCompletionUsage | None = None,
) -> _OpenAIResponseObjectWithInputAndMessages:
    from ogx_api import OpenAIAssistantMessageParam, OpenAIUserMessageParam
    from ogx_api.openai_responses import OpenAIResponseUsage

    return _OpenAIResponseObjectWithInputAndMessages(
        id="resp-previous-123",
        object="response",
        created_at=1234567890,
        model="meta-llama/Llama-3.1-8B-Instruct",
        status="completed",
        text=OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
        input=[
            OpenAIResponseMessage(
                id="msg-prev-user",
                role="user",
                content=[OpenAIResponseInputMessageContentText(text=input_text)],
            )
        ],
        output=[
            OpenAIResponseMessage(
                id="msg-prev-assistant",
                role="assistant",
                content=[OpenAIResponseOutputMessageContentOutputText(text=output_text)],
            )
        ],
        messages=[
            OpenAIUserMessageParam(content=input_text),
            OpenAIAssistantMessageParam(content=output_text),
        ],
        usage=OpenAIResponseUsage(
            input_tokens=5,
            output_tokens=5,
            total_tokens=10,
            input_tokens_details={"cached_tokens": 0},
            output_tokens_details={"reasoning_tokens": 0},
        )
        if usage is None
        else usage,
        store=True,
    )
