# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Awaitable, Callable

import tiktoken

from ogx.core.datatypes import CompressionConfig
from ogx.log import get_logger
from ogx_api import (
    InvalidParameterError,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)

logger = get_logger(name=__name__, category="inference")

# Used only when tiktoken cannot resolve an encoding for the model.
APPROX_CHARS_PER_TOKEN = 4

DUPLICATE_TOOL_OUTPUT_PLACEHOLDER = "[duplicate tool output omitted to save tokens]"

SummarizeCallback = Callable[[OpenAIChatCompletionRequestWithExtraBody], Awaitable[OpenAIChatCompletion]]


def _message_text(message: OpenAIMessageParam) -> str:
    """Flatten a message's content to plain text, skipping non-text parts (e.g. images)."""
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(part.text for part in content if hasattr(part, "text"))


def resolve_encoding(model: str, config: CompressionConfig, extra_body: dict | None = None) -> tiktoken.Encoding | None:
    """Resolve tiktoken encoding via a 5-step chain. Returns None for character fallback.

    Mirrors OpenAIResponsesImpl._resolve_encoding, adapted to take a CompressionConfig
    instead of instance state.
    """
    # 1. Per-request override (fail hard if invalid)
    if extra_body and (enc_name := extra_body.get("tokenizer_encoding")):
        try:
            return tiktoken.get_encoding(enc_name)
        except ValueError:
            raise InvalidParameterError(
                "tokenizer_encoding",
                enc_name,
                "Must be a valid tiktoken encoding name (e.g. 'o200k_base', 'cl100k_base').",
            ) from None

    # 2. Admin default
    if config.tokenizer_encoding:
        return tiktoken.get_encoding(config.tokenizer_encoding)

    # 3. tiktoken built-in (soft fail)
    model_name = model.split("/")[-1] if "/" in model else model
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        pass

    # 4. Model-family mapping (soft fail)
    base = model_name.lower()
    for prefix, enc_name in config.model_tokenizer_mappings.items():
        if base.startswith(prefix.lower()):
            try:
                return tiktoken.get_encoding(enc_name)
            except ValueError:
                logger.warning("Invalid encoding in model_tokenizer_mappings", prefix=prefix, encoding=enc_name)
                break

    # 5. Character fallback
    logger.debug("Could not resolve tokenizer encoding, using character-based estimate", model=model)
    return None


def count_message_tokens(
    messages: list[OpenAIMessageParam], model: str, config: CompressionConfig, extra_body: dict | None = None
) -> int:
    """Estimate the token count of a list of messages. Uses tiktoken when possible, a
    character-based estimate as fallback."""
    encoding = resolve_encoding(model, config, extra_body)
    texts = [_message_text(m) for m in messages]
    if encoding is not None:
        return sum(len(encoding.encode(t)) for t in texts if t)
    total_chars = sum(len(t) for t in texts)
    return max(1, total_chars // APPROX_CHARS_PER_TOKEN)


def dedupe_tool_outputs(messages: list[OpenAIMessageParam]) -> list[OpenAIMessageParam]:
    """Collapse exact-duplicate tool-call outputs to a short placeholder.

    Never removes messages outright: providers require every tool_call to be immediately
    followed by a matching tool response, so dropping a duplicate message would break that
    pairing. Replacing its content is safe and still saves the bulk of the tokens.
    """
    seen_content: set[str] = set()
    result: list[OpenAIMessageParam] = []
    for message in messages:
        if isinstance(message, OpenAIToolMessageParam):
            text = _message_text(message)
            if text and text in seen_content:
                message = message.model_copy(update={"content": DUPLICATE_TOOL_OUTPUT_PLACEHOLDER})
            else:
                seen_content.add(text)
        result.append(message)
    return result


def truncate_oversized_tool_outputs(
    messages: list[OpenAIMessageParam], model: str, config: CompressionConfig
) -> list[OpenAIMessageParam]:
    """Truncate any single tool output exceeding config.max_tool_output_tokens."""
    if not config.max_tool_output_tokens:
        return messages

    encoding = resolve_encoding(model, config)
    result: list[OpenAIMessageParam] = []
    for message in messages:
        if isinstance(message, OpenAIToolMessageParam):
            text = _message_text(message)
            token_count = len(encoding.encode(text)) if encoding is not None else len(text) // APPROX_CHARS_PER_TOKEN
            if token_count > config.max_tool_output_tokens:
                if encoding is not None:
                    truncated = encoding.decode(encoding.encode(text)[: config.max_tool_output_tokens])
                else:
                    truncated = text[: config.max_tool_output_tokens * APPROX_CHARS_PER_TOKEN]
                message = message.model_copy(update={"content": truncated + "\n[tool output truncated]"})
        result.append(message)
    return result


def _split_into_turns(
    messages: list[OpenAIMessageParam],
) -> tuple[list[OpenAIMessageParam], list[list[OpenAIMessageParam]]]:
    """Split messages into (leading system/developer messages, list of turns).

    A turn starts at each user-role message and includes every following non-user message
    (assistant/tool) up to, but not including, the next user message. This keeps an
    assistant's tool_calls together with their tool responses, since dropping one without
    the other would violate the chat completion API contract.
    """
    leading: list[OpenAIMessageParam] = []
    turns: list[list[OpenAIMessageParam]] = []
    for message in messages:
        if message.role == "user":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
        else:
            # No user message seen yet: treat as part of the leading (always-kept) prefix.
            leading.append(message)
    return leading, turns


def _build_summarization_request(
    turns: list[list[OpenAIMessageParam]], model: str, config: CompressionConfig
) -> OpenAIChatCompletionRequestWithExtraBody:
    flattened = [message for turn in turns for message in turn]
    texts = [_message_text(m) for m in flattened if _message_text(m)]
    transcript = "\n\n".join(texts)
    return OpenAIChatCompletionRequestWithExtraBody(
        model=config.summarization_model or model,
        messages=[
            OpenAISystemMessageParam(content=config.summarization_prompt),
            OpenAIUserMessageParam(content=transcript),
        ],
        stream=False,
    )


async def _window_and_maybe_summarize(
    leading: list[OpenAIMessageParam],
    turns: list[list[OpenAIMessageParam]],
    model: str,
    config: CompressionConfig,
    summarize: SummarizeCallback | None,
) -> list[OpenAIMessageParam]:
    """Drop the oldest turns until under config.max_context_tokens, always keeping at least
    the most recent turn. Optionally summarizes dropped turns via `summarize` instead of
    silently dropping them."""
    assert config.max_context_tokens is not None

    dropped: list[list[OpenAIMessageParam]] = []
    kept = list(turns)
    while len(kept) > 1 and count_message_tokens(leading + [m for t in kept for m in t], model, config) > (
        config.max_context_tokens
    ):
        dropped.append(kept.pop(0))

    if not dropped:
        return leading + [m for t in kept for m in t]

    if summarize is not None:
        request = _build_summarization_request(dropped, model, config)
        try:
            response = await summarize(request)
            summary_text = response.choices[0].message.content or ""
            summary_message: OpenAIMessageParam = OpenAISystemMessageParam(
                content=f"[Summary of earlier conversation turns]\n{summary_text}"
            )
            return leading + [summary_message] + [m for t in kept for m in t]
        except Exception as e:
            logger.warning("Failed to summarize dropped conversation turns, dropping without summary", error=str(e))

    return leading + [m for t in kept for m in t]


async def compress_messages(
    messages: list[OpenAIMessageParam],
    model: str,
    config: CompressionConfig,
    summarize: SummarizeCallback | None,
) -> list[OpenAIMessageParam]:
    """Apply dedup, truncation, and (if over budget) windowing/summarization to a message list."""
    if config.dedupe_tool_outputs:
        messages = dedupe_tool_outputs(messages)
    if config.max_tool_output_tokens:
        messages = truncate_oversized_tool_outputs(messages, model, config)

    if config.max_context_tokens is not None and count_message_tokens(messages, model, config) > (
        config.max_context_tokens
    ):
        leading, turns = _split_into_turns(messages)
        if turns:
            messages = await _window_and_maybe_summarize(leading, turns, model, config, summarize)

    return messages


def apply_output_token_cap(params: OpenAIChatCompletionRequestWithExtraBody, config: CompressionConfig) -> None:
    """Clamp/default max_tokens and max_completion_tokens to config.max_output_tokens."""
    if not config.max_output_tokens:
        return
    if params.max_tokens is None and params.max_completion_tokens is None:
        params.max_completion_tokens = config.max_output_tokens
        return
    if params.max_tokens is not None:
        params.max_tokens = min(params.max_tokens, config.max_output_tokens)
    if params.max_completion_tokens is not None:
        params.max_completion_tokens = min(params.max_completion_tokens, config.max_output_tokens)


async def maybe_compress_request(
    params: OpenAIChatCompletionRequestWithExtraBody,
    config: CompressionConfig | None,
    model_id: str,
    summarize: SummarizeCallback | None,
) -> None:
    """Mutate `params` in place: cap output tokens and compress `params.messages` when
    compression is enabled. No-op when config is None or disabled."""
    if config is None or not config.enabled:
        return

    apply_output_token_cap(params, config)
    params.messages = await compress_messages(params.messages, model_id, config, summarize)
