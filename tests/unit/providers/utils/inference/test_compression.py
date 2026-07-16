# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

from ogx.core.datatypes import CompressionConfig
from ogx.providers.utils.inference.compression import (
    _split_into_turns,
    apply_output_token_cap,
    compress_messages,
    count_message_tokens,
    dedupe_tool_outputs,
    maybe_compress_request,
    resolve_encoding,
    truncate_oversized_tool_outputs,
)
from ogx_api import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)


def _tool_call(call_id: str) -> OpenAIAssistantMessageParam:
    return OpenAIAssistantMessageParam(
        content=None,
        tool_calls=[
            OpenAIChatCompletionToolCall(
                id=call_id, function=OpenAIChatCompletionToolCallFunction(name="get_weather", arguments="{}")
            )
        ],
    )


class TestResolveEncoding:
    def test_resolves_known_model_via_tiktoken(self):
        config = CompressionConfig()
        encoding = resolve_encoding("gpt-4o", config)
        assert encoding is not None

    def test_admin_default_takes_precedence(self):
        config = CompressionConfig(tokenizer_encoding="cl100k_base")
        encoding = resolve_encoding("totally-unknown-model-xyz", config)
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_falls_back_to_model_family_mapping(self):
        config = CompressionConfig()
        encoding = resolve_encoding("ollama/llama3.2:3b", config)
        assert encoding is not None

    def test_returns_none_for_unresolvable_model(self):
        config = CompressionConfig(model_tokenizer_mappings={})
        encoding = resolve_encoding("some-totally-unknown-model-xyz-123", config)
        assert encoding is None


class TestCountMessageTokens:
    def test_counts_more_tokens_for_longer_text(self):
        config = CompressionConfig()
        short = count_message_tokens([OpenAIUserMessageParam(content="hi")], "gpt-4o", config)
        long = count_message_tokens([OpenAIUserMessageParam(content="hi " * 200)], "gpt-4o", config)
        assert long > short

    def test_char_fallback_when_encoding_unresolvable(self):
        config = CompressionConfig(model_tokenizer_mappings={})
        count = count_message_tokens([OpenAIUserMessageParam(content="a" * 40)], "unresolvable-model-xyz", config)
        assert count == 10  # 40 chars / APPROX_CHARS_PER_TOKEN (4)


class TestDedupeToolOutputs:
    def test_collapses_exact_duplicate_tool_output(self):
        messages = [
            OpenAIUserMessageParam(content="weather?"),
            _tool_call("call_1"),
            OpenAIToolMessageParam(tool_call_id="call_1", content="sunny, 75F"),
            OpenAIUserMessageParam(content="and tomorrow?"),
            _tool_call("call_2"),
            OpenAIToolMessageParam(tool_call_id="call_2", content="sunny, 75F"),
        ]

        result = dedupe_tool_outputs(messages)

        assert result[2].content == "sunny, 75F"
        assert result[5].content == "[duplicate tool output omitted to save tokens]"

    def test_leaves_distinct_tool_outputs_untouched(self):
        messages = [
            OpenAIToolMessageParam(tool_call_id="call_1", content="sunny"),
            OpenAIToolMessageParam(tool_call_id="call_2", content="rainy"),
        ]

        result = dedupe_tool_outputs(messages)

        assert result[0].content == "sunny"
        assert result[1].content == "rainy"


class TestTruncateOversizedToolOutputs:
    def test_truncates_output_exceeding_budget(self):
        config = CompressionConfig(max_tool_output_tokens=5)
        messages = [OpenAIToolMessageParam(tool_call_id="call_1", content="word " * 100)]

        result = truncate_oversized_tool_outputs(messages, "gpt-4o", config)

        assert result[0].content.endswith("[tool output truncated]")
        assert len(result[0].content) < len(messages[0].content)

    def test_leaves_small_output_untouched(self):
        config = CompressionConfig(max_tool_output_tokens=1000)
        messages = [OpenAIToolMessageParam(tool_call_id="call_1", content="short output")]

        result = truncate_oversized_tool_outputs(messages, "gpt-4o", config)

        assert result[0].content == "short output"

    def test_noop_when_max_tool_output_tokens_unset(self):
        config = CompressionConfig(max_tool_output_tokens=None)
        messages = [OpenAIToolMessageParam(tool_call_id="call_1", content="word " * 10000)]

        result = truncate_oversized_tool_outputs(messages, "gpt-4o", config)

        assert result == messages


class TestSplitIntoTurns:
    def test_keeps_system_messages_as_leading_prefix(self):
        messages = [
            OpenAISystemMessageParam(content="be nice"),
            OpenAIUserMessageParam(content="hi"),
            OpenAIAssistantMessageParam(content="hello"),
        ]

        leading, turns = _split_into_turns(messages)

        assert [m.role for m in leading] == ["system"]
        assert len(turns) == 1
        assert [m.role for m in turns[0]] == ["user", "assistant"]

    def test_keeps_tool_call_and_response_in_same_turn(self):
        messages = [
            OpenAIUserMessageParam(content="weather?"),
            _tool_call("call_1"),
            OpenAIToolMessageParam(tool_call_id="call_1", content="sunny"),
            OpenAIUserMessageParam(content="thanks"),
        ]

        leading, turns = _split_into_turns(messages)

        assert leading == []
        assert len(turns) == 2
        assert [m.role for m in turns[0]] == ["user", "assistant", "tool"]
        assert [m.role for m in turns[1]] == ["user"]


class TestCompressMessages:
    async def test_noop_below_budget(self):
        config = CompressionConfig(max_context_tokens=10_000)
        messages = [
            OpenAISystemMessageParam(content="be nice"),
            OpenAIUserMessageParam(content="hi"),
        ]

        result = await compress_messages(messages, "gpt-4o", config, summarize=None)

        assert len(result) == 2

    async def test_windows_out_oldest_turns_when_over_budget(self):
        config = CompressionConfig(max_context_tokens=20, dedupe_tool_outputs=False)
        messages = [
            OpenAISystemMessageParam(content="be nice"),
            OpenAIUserMessageParam(content="turn one " * 20),
            OpenAIAssistantMessageParam(content="reply one " * 20),
            OpenAIUserMessageParam(content="turn two"),
        ]

        result = await compress_messages(messages, "gpt-4o", config, summarize=None)

        # Oldest turn (turn one + reply one) should be dropped; system + last turn remain.
        assert [m.role for m in result] == ["system", "user"]
        assert result[1].content == "turn two"

    async def test_always_keeps_at_least_the_most_recent_turn(self):
        config = CompressionConfig(max_context_tokens=1)
        messages = [
            OpenAIUserMessageParam(content="turn one"),
            OpenAIUserMessageParam(content="turn two " * 50),
        ]

        result = await compress_messages(messages, "gpt-4o", config, summarize=None)

        assert len(result) == 1
        assert "turn two" in result[0].content

    async def test_summarizes_dropped_turns_when_summarize_provided(self):
        config = CompressionConfig(max_context_tokens=20, summarize_dropped_turns=True)
        messages = [
            OpenAIUserMessageParam(content="turn one " * 20),
            OpenAIAssistantMessageParam(content="reply one " * 20),
            OpenAIUserMessageParam(content="turn two"),
        ]

        async def fake_summarize(request: OpenAIChatCompletionRequestWithExtraBody) -> OpenAIChatCompletion:
            return OpenAIChatCompletion(
                id="resp_1",
                model=request.model,
                created=int(time.time()),
                choices=[
                    OpenAIChoice(
                        index=0,
                        finish_reason="stop",
                        message=OpenAIChatCompletionResponseMessage(content="short summary"),
                    )
                ],
            )

        result = await compress_messages(messages, "gpt-4o", config, summarize=fake_summarize)

        assert result[0].role == "system"
        assert "short summary" in result[0].content
        assert result[1].content == "turn two"

    async def test_falls_back_to_drop_when_summarize_raises(self):
        config = CompressionConfig(max_context_tokens=20, summarize_dropped_turns=True)
        messages = [
            OpenAIUserMessageParam(content="turn one " * 20),
            OpenAIUserMessageParam(content="turn two"),
        ]

        async def failing_summarize(request: OpenAIChatCompletionRequestWithExtraBody) -> OpenAIChatCompletion:
            raise RuntimeError("boom")

        result = await compress_messages(messages, "gpt-4o", config, summarize=failing_summarize)

        assert [m.content for m in result] == ["turn two"]


class TestApplyOutputTokenCap:
    def test_sets_default_when_unset(self):
        config = CompressionConfig(max_output_tokens=100)
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")]
        )

        apply_output_token_cap(params, config)

        assert params.max_completion_tokens == 100
        assert params.max_tokens is None

    def test_clamps_value_above_cap(self):
        config = CompressionConfig(max_output_tokens=100)
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")], max_tokens=500
        )

        apply_output_token_cap(params, config)

        assert params.max_tokens == 100

    def test_leaves_value_below_cap_untouched(self):
        config = CompressionConfig(max_output_tokens=1000)
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")], max_completion_tokens=50
        )

        apply_output_token_cap(params, config)

        assert params.max_completion_tokens == 50

    def test_noop_when_cap_unset(self):
        config = CompressionConfig(max_output_tokens=None)
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")]
        )

        apply_output_token_cap(params, config)

        assert params.max_tokens is None
        assert params.max_completion_tokens is None


class TestMaybeCompressRequest:
    async def test_noop_when_config_is_none(self):
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")]
        )
        original_messages = list(params.messages)

        await maybe_compress_request(params, None, model_id="gpt-4o", summarize=None)

        assert params.messages == original_messages

    async def test_noop_when_disabled(self):
        config = CompressionConfig(enabled=False, max_output_tokens=1)
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")]
        )

        await maybe_compress_request(params, config, model_id="gpt-4o", summarize=None)

        assert params.max_completion_tokens is None

    async def test_applies_compression_when_enabled(self):
        config = CompressionConfig(enabled=True, max_output_tokens=42)
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4o", messages=[OpenAIUserMessageParam(content="hi")]
        )

        await maybe_compress_request(params, config, model_id="gpt-4o", summarize=None)

        assert params.max_completion_tokens == 42
