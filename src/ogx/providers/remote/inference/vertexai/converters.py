# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pure translation functions between OpenAI format and google-genai native API.

No SDK calls or side effects — only type conversions.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ogx.log import get_logger
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionCustomToolCall,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAIChoiceDelta,
    OpenAIChoiceLogprobs,
    OpenAIChunkChoice,
    OpenAICompletion,
    OpenAICompletionChoice,
    OpenAIFinishReason,
    OpenAITokenLogProb,
    OpenAITopLogProb,
)
from ogx_api.inference.models import (
    OpenAIChatCompletionUsageCompletionTokensDetails,
    OpenAIChatCompletionUsagePromptTokensDetails,
)

logger = get_logger(__name__, category="inference")

if TYPE_CHECKING:
    from google.genai import types as genai_types


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert *obj* to a plain dict (handles Pydantic models, dicts, and other objects)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        result: dict[str, Any] = obj.model_dump()
        return result
    return dict(obj)


class _GeminiLogprobCandidate(BaseModel):
    """Typed wrapper for a single Gemini logprob candidate (chosen or top-K)."""

    model_config = ConfigDict(from_attributes=True)

    token: str = ""
    log_probability: float | None = None
    token_id: int | None = None

    @field_validator("token", mode="before")
    @classmethod
    def _coerce_none_token(cls, v: Any) -> str:
        return v if v is not None else ""


class _GeminiTopCandidatesEntry(BaseModel):
    """Typed wrapper for a top_candidates entry (parallel to chosen_candidates)."""

    model_config = ConfigDict(from_attributes=True)

    candidates: list[_GeminiLogprobCandidate] = []

    @field_validator("candidates", mode="before")
    @classmethod
    def _coerce_none_list(cls, v: Any) -> list:
        return v if v is not None else []


class _GeminiLogprobsResult(BaseModel):
    """Typed wrapper for a Gemini logprobs_result object."""

    model_config = ConfigDict(from_attributes=True)

    chosen_candidates: list[_GeminiLogprobCandidate] = []
    top_candidates: list[_GeminiTopCandidatesEntry] = []

    @field_validator("chosen_candidates", "top_candidates", mode="before")
    @classmethod
    def _coerce_none_list(cls, v: Any) -> list:
        return v if v is not None else []


@dataclass(frozen=True)
class _CandidateData:
    """Processed candidate data shared between response and stream converters."""

    index: int
    text: str | None
    reasoning_content: str | None  # thinking text from part.thought
    tool_calls: list[OpenAIChatCompletionToolCall | OpenAIChatCompletionCustomToolCall]
    finish_reason_raw: Any
    logprobs: OpenAIChoiceLogprobs | None


_GEMINI_TO_OPENAI_FINISH_REASON: dict[str, OpenAIFinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "LANGUAGE": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
}


def convert_finish_reason(
    finish_reason: str | None,
) -> OpenAIFinishReason:
    """Map a Gemini FinishReason string to the OpenAI finish_reason literal."""
    if finish_reason is None:
        return "stop"
    reason_str = str(finish_reason).upper()
    return _GEMINI_TO_OPENAI_FINISH_REASON.get(reason_str, "stop")


def convert_response_format(
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert an OpenAI ``response_format`` parameter to google-genai config kwargs.

    Returns a dict that can be merged into ``GenerateContentConfig`` kwargs.
    Supports ``json_object`` → ``response_mime_type='application/json'``
    and ``json_schema`` → ``response_mime_type='application/json'`` + ``response_schema``.
    """
    if response_format is None:
        return {}

    fmt_type = response_format.get("type")
    if fmt_type == "json_object":
        return {"response_mime_type": "application/json"}
    if fmt_type == "json_schema":
        result: dict[str, Any] = {"response_mime_type": "application/json"}
        json_schema = response_format.get("json_schema")
        if json_schema and isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if schema:
                result["response_schema"] = schema
        return result
    # "text" or unknown → no special config
    return {}


def _extract_text_content(content: str | list[dict[str, Any]] | None) -> str:
    """Extract plain text from OpenAI message content (string or content parts list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # content is a list of content parts
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
        elif isinstance(part, str):
            texts.append(part)
        elif hasattr(part, "type") and part.type == "text":
            texts.append(getattr(part, "text", ""))
    return "".join(texts)


def _convert_image_url_part(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI image_url content part to a Gemini inline_data part.

    Returns None for unsupported URL schemes (for example: file://, ftp://, gs://).
    """
    image_url = part.get("image_url", {})
    url = image_url.get("url", "") if isinstance(image_url, dict) else getattr(image_url, "url", "")

    match = re.match(r"data:image/(\w+);base64,(.+)", url)
    if match:
        fmt, image_data = match.groups()
        content = base64.b64decode(image_data)
        return {"inline_data": {"data": content, "mime_type": f"image/{fmt}"}}

    if url.startswith(("http://", "https://")):
        logger.warning("HTTP image URL reached converter without pre-download; skipping")
        return None

    logger.warning("Unsupported image URL scheme in user message, skipping", url=url[:50])
    return None


def _convert_user_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI user message to a Gemini Content dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        parts = [{"text": content}]
    else:
        parts = []
        for part in content:
            part_dict = _to_dict(part)
            part_type = part_dict.get("type")
            if part_type == "text":
                text = part_dict.get("text", "")
                if text:
                    parts.append({"text": text})
            elif part_type == "image_url":
                inline = _convert_image_url_part(part_dict)
                if inline is not None:
                    parts.append(inline)
            else:
                logger.warning(
                    "Unsupported content part type '%s' in user message; skipping",
                    part_type,
                )

    return {"role": "user", "parts": parts}


def _parse_tool_call_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments from string or dict form."""
    if not isinstance(arguments, str):
        return arguments if isinstance(arguments, dict) else {}
    try:
        parsed: dict[str, Any] = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed


def _convert_assistant_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI assistant message to a Gemini Content dict.

    Returns ``None`` when the message has no text content and no tool calls.
    """
    parts: list[dict[str, Any]] = []

    text = _extract_text_content(msg.get("content"))
    if text:
        parts.append({"text": text})

    for tc in msg.get("tool_calls") or []:
        tc = _to_dict(tc)
        func = tc.get("function", {})
        parts.append(
            {
                "function_call": {
                    "name": func.get("name", ""),
                    "args": _parse_tool_call_arguments(func.get("arguments", "{}")),
                }
            }
        )

    return {"role": "model", "parts": parts} if parts else None


def _convert_tool_message(msg: dict[str, Any], all_messages: list[Any]) -> dict[str, Any]:
    """Convert an OpenAI tool-result message to a Gemini Content dict."""
    tool_call_id = msg.get("tool_call_id", "")
    tool_content = _extract_text_content(msg.get("content"))
    try:
        response_data = json.loads(tool_content)
    except (json.JSONDecodeError, TypeError):
        response_data = {"result": tool_content}
    if not isinstance(response_data, dict):
        response_data = {"result": response_data}

    func_name = _find_function_name_for_tool_call_id(all_messages, tool_call_id)
    return {
        "role": "user",
        "parts": [{"function_response": {"name": func_name, "response": response_data}}],
    }


def convert_openai_messages_to_gemini(
    messages: list[Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Gemini Content dicts.

    Returns ``(system_instruction, contents)`` where:
    - ``system_instruction`` is extracted from system/developer messages (or ``None``).
    - ``contents`` is a list of Gemini ``Content``-like dicts with ``role`` and ``parts``.

    Gemini uses ``"user"`` and ``"model"`` roles (no ``"assistant"``).
    Tool results use ``"user"`` role with ``function_response`` parts.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for raw_msg in messages:
        msg = _to_dict(raw_msg)
        role = msg.get("role", "")

        if role in ("system", "developer"):
            text = _extract_text_content(msg.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            contents.append(_convert_user_message(msg))
        elif role == "assistant":
            converted = _convert_assistant_message(msg)
            if converted:
                contents.append(converted)
        elif role == "tool":
            contents.append(_convert_tool_message(msg, messages))

    system_instruction = "\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _find_function_name_for_tool_call_id(messages: list[Any], tool_call_id: str) -> str:
    """Search through messages for the function name matching a tool_call_id."""
    for msg in messages:
        msg = _to_dict(msg)
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    tc = _to_dict(tc)
                    if tc.get("id") == tool_call_id:
                        func = tc.get("function", {})
                        return str(func.get("name", "unknown"))
    return "unknown"


def convert_openai_tools_to_gemini(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI tools array to Gemini Tool format.

    OpenAI format::

        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]

    Gemini format::

        [{"function_declarations": [{"name": ..., "description": ..., "parameters": {...}}]}]

    Returns ``None`` if no tools are provided.
    """
    if not tools:
        return None

    function_declarations: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        decl: dict[str, Any] = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if params:
            decl["parameters_json_schema"] = params
        function_declarations.append(decl)

    if not function_declarations:
        return None

    return [{"function_declarations": function_declarations}]


def convert_deprecated_functions_to_tools(
    functions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert deprecated OpenAI ``functions`` parameter to the modern ``tools`` format.

    Each function dict is wrapped in ``{"type": "function", "function": func}``
    to match the shape expected by ``convert_openai_tools_to_gemini()``.
    """
    return [{"type": "function", "function": func} for func in functions]


def convert_deprecated_function_call_to_tool_choice(
    function_call: str | dict[str, Any],
) -> str | dict[str, Any]:
    """Convert deprecated OpenAI ``function_call`` parameter to the modern ``tool_choice`` format.

    Mapping:
    - ``"auto"`` → ``"auto"``
    - ``"none"`` → ``"none"``
    - ``{"name": "fn_name"}`` → ``{"type": "function", "function": {"name": "fn_name"}}``
    """
    if isinstance(function_call, str):
        return function_call  # "auto" or "none" pass through unchanged
    if isinstance(function_call, dict) and "name" in function_call:
        return {"type": "function", "function": {"name": function_call["name"]}}
    return function_call  # fallback: return as-is


def generate_completion_id() -> str:
    """Generate a unique completion ID in OpenAI format."""
    return f"chatcmpl-{uuid.uuid4()}"


def _extract_candidate_parts(
    candidate: Any,
) -> tuple[list[str], list[str], list[OpenAIChatCompletionToolCall | OpenAIChatCompletionCustomToolCall]]:
    """Extract text segments, thinking segments, and tool calls from a Gemini candidate's parts."""
    content_obj = getattr(candidate, "content", None)
    parts = getattr(content_obj, "parts", None) or []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[OpenAIChatCompletionToolCall | OpenAIChatCompletionCustomToolCall] = []

    for part in parts:
        # Check for thinking/reasoning content first (part.thought is a bool flag;
        # the actual thinking text is in part.text).
        thought = getattr(part, "thought", None)
        if thought:
            part_text = getattr(part, "text", None)
            if part_text is not None:
                thinking_parts.append(part_text)
            continue

        part_text = getattr(part, "text", None)
        if part_text is not None:
            text_parts.append(part_text)
            continue

        fc = getattr(part, "function_call", None)
        if fc is not None:
            tool_calls.append(
                OpenAIChatCompletionToolCall(
                    index=len(tool_calls),
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=OpenAIChatCompletionToolCallFunction(
                        name=getattr(fc, "name", "") or "",
                        arguments=json.dumps(getattr(fc, "args", {}) or {}),
                    ),
                )
            )

    return text_parts, thinking_parts, tool_calls


def _build_top_logprobs(
    top_candidates_list: list[_GeminiTopCandidatesEntry],
    index: int,
) -> list[OpenAITopLogProb] | None:
    """Build top logprobs list from Gemini top_candidates at a given index.

    Returns None if index is out of bounds, otherwise returns list of OpenAITopLogProb.
    """
    if index >= len(top_candidates_list):
        return None

    return [
        OpenAITopLogProb(
            token=c.token,
            bytes=list(c.token.encode("utf-8")) if c.token else None,
            logprob=c.log_probability if c.log_probability is not None else 0.0,
        )
        for c in top_candidates_list[index].candidates
    ]


def _build_token_logprob(
    chosen_cand: _GeminiLogprobCandidate,
    top_candidates_list: list[_GeminiTopCandidatesEntry],
    index: int,
) -> OpenAITokenLogProb | None:
    """Build a single token logprob entry from a chosen candidate.

    Returns None if log_probability is None (token should be skipped).
    """
    if chosen_cand.log_probability is None:
        return None

    token_bytes = list(chosen_cand.token.encode("utf-8")) if chosen_cand.token else None

    return OpenAITokenLogProb(
        token=chosen_cand.token,
        bytes=token_bytes,
        logprob=chosen_cand.log_probability,
        top_logprobs=_build_top_logprobs(top_candidates_list, index),
    )


def _extract_logprobs(candidate: Any) -> OpenAIChoiceLogprobs | None:
    """Extract log probability data from a Gemini candidate and convert to OpenAI format.

    Gemini's logprobs_result contains:
    - chosen_candidates: the actual tokens selected at each position
    - top_candidates: the top-K alternative tokens at each position (parallel list)
    """
    raw_logprobs = getattr(candidate, "logprobs_result", None)
    if raw_logprobs is None:
        return None

    parsed = _GeminiLogprobsResult.model_validate(raw_logprobs)

    token_logprobs: list[OpenAITokenLogProb] = []
    for i, chosen_cand in enumerate(parsed.chosen_candidates):
        token_logprob = _build_token_logprob(chosen_cand, parsed.top_candidates, i)
        if token_logprob is not None:
            token_logprobs.append(token_logprob)

    if not token_logprobs:
        return None
    return OpenAIChoiceLogprobs(content=token_logprobs)


def _process_candidates(response_or_chunk: Any) -> list[_CandidateData]:
    """Extract and process all candidates from a Gemini response or streaming chunk."""
    result: list[_CandidateData] = []
    for i, candidate in enumerate(getattr(response_or_chunk, "candidates", None) or []):
        text_parts, thinking_parts, tool_calls = _extract_candidate_parts(candidate)
        result.append(
            _CandidateData(
                index=i,
                text="".join(text_parts) if text_parts else None,
                reasoning_content="".join(thinking_parts) if thinking_parts else None,
                tool_calls=tool_calls,
                finish_reason_raw=getattr(candidate, "finish_reason", None),
                logprobs=_extract_logprobs(candidate),
            )
        )
    return result


def _resolve_finish_reason_common(
    finish_reason_val: Any,
    has_tool_calls: bool,
    *,
    allow_none: bool,
) -> OpenAIFinishReason | None:
    if has_tool_calls:
        return "tool_calls"
    if finish_reason_val is None:
        return None if allow_none else "stop"
    return convert_finish_reason(str(finish_reason_val))


def _resolve_finish_reason(
    finish_reason_val: Any,
    has_tool_calls: bool,
) -> OpenAIFinishReason:
    """Determine the OpenAI finish reason for a candidate."""
    finish_reason = _resolve_finish_reason_common(finish_reason_val, has_tool_calls, allow_none=False)
    return "stop" if finish_reason is None else finish_reason


def convert_gemini_response_to_openai(
    response: genai_types.GenerateContentResponse,
    model: str,
) -> OpenAIChatCompletion:
    """Map a google-genai ``GenerateContentResponse`` to ``OpenAIChatCompletion``."""
    completion_id = generate_completion_id()
    created = int(time.time())

    choices: list[OpenAIChoice] = []
    for cd in _process_candidates(response):
        # NOTE: reasoning_content is only available on streaming deltas (OpenAIChoiceDelta).
        # Non-streaming responses include thinking text in content because
        # OpenAIChatCompletionResponseMessage has no reasoning_content field.
        content_parts = []
        if cd.reasoning_content:
            content_parts.append(cd.reasoning_content)
        if cd.text:
            content_parts.append(cd.text)
        content = "".join(content_parts) if content_parts else None

        choices.append(
            OpenAIChoice(
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content=content,
                    tool_calls=cd.tool_calls or None,
                ),
                finish_reason=_resolve_finish_reason(cd.finish_reason_raw, bool(cd.tool_calls)),
                index=cd.index,
                logprobs=cd.logprobs,
            )
        )

    if not choices:
        choices.append(
            OpenAIChoice(
                message=OpenAIChatCompletionResponseMessage(role="assistant", content=None),
                finish_reason="content_filter",
                index=0,
            )
        )

    return OpenAIChatCompletion(
        id=completion_id,
        choices=choices,
        created=created,
        model=model,
        usage=extract_usage(response),
    )


def extract_usage(
    response: genai_types.GenerateContentResponse,
) -> OpenAIChatCompletionUsage | None:
    """Extract token usage from a Gemini response."""
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta is None:
        return None

    prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage_meta, "total_token_count", 0) or 0

    cached_tokens = getattr(usage_meta, "cached_content_token_count", None)
    prompt_details = None
    if cached_tokens:
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=cached_tokens)

    reasoning_tokens = getattr(usage_meta, "thoughts_token_count", None)
    completion_details = None
    if reasoning_tokens:
        completion_details = OpenAIChatCompletionUsageCompletionTokensDetails(reasoning_tokens=reasoning_tokens)

    return OpenAIChatCompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
    )


def _resolve_stream_finish_reason(
    finish_reason_val: Any,
    has_tool_calls: bool,
) -> OpenAIFinishReason | None:
    """Determine the OpenAI finish reason for a streaming chunk candidate.

    Unlike the non-streaming variant, returns ``None`` when the candidate has
    no finish reason yet (mid-stream).
    """
    return _resolve_finish_reason_common(finish_reason_val, has_tool_calls, allow_none=True)


def convert_gemini_stream_chunk_to_openai(
    chunk: genai_types.GenerateContentResponse,
    model: str,
    completion_id: str,
    is_first_chunk: bool,
) -> OpenAIChatCompletionChunk:
    """Map a Gemini streaming chunk to ``OpenAIChatCompletionChunk``."""
    created = int(time.time())
    role: Literal["assistant"] | None = "assistant" if is_first_chunk else None

    choices: list[OpenAIChunkChoice] = [
        OpenAIChunkChoice(
            delta=OpenAIChoiceDelta(
                role=role,
                content=cd.text,
                tool_calls=cd.tool_calls or None,  # type: ignore[arg-type]
                reasoning_content=cd.reasoning_content,
            ),
            finish_reason=_resolve_stream_finish_reason(cd.finish_reason_raw, bool(cd.tool_calls)),
            index=cd.index,
            logprobs=cd.logprobs,
        )
        for cd in _process_candidates(chunk)
    ]

    if not choices:
        choices.append(
            OpenAIChunkChoice(
                delta=OpenAIChoiceDelta(role=role, content=None),
                finish_reason=None,
                index=0,
            )
        )

    return OpenAIChatCompletionChunk(
        id=completion_id,
        choices=choices,
        created=created,
        model=model,
        usage=extract_usage(chunk),
    )


def convert_gemini_stream_chunk_to_openai_completion(
    chunk: genai_types.GenerateContentResponse,
    model: str,
    completion_id: str,
    index_offset: int = 0,
) -> OpenAICompletion:
    """Map a Gemini streaming chunk to ``OpenAICompletion`` (text completions)."""
    candidates = getattr(chunk, "candidates", None) or []

    choices: list[OpenAICompletionChoice] = []
    for i, candidate in enumerate(candidates):
        text = ""
        if candidate.content and candidate.content.parts:
            text = getattr(candidate.content.parts[0], "text", "") or ""
        finish_reason_val = getattr(candidate, "finish_reason", None)
        finish_reason = _resolve_stream_finish_reason(finish_reason_val, has_tool_calls=False)
        # finish_reason can be None mid-stream; OpenAICompletionChoice requires a value.
        # Use "stop" as default for mid-stream chunks since the field is required.
        if finish_reason is None:
            finish_reason = "stop"
        choices.append(
            OpenAICompletionChoice(
                text=text,
                finish_reason=finish_reason,
                index=i + index_offset,
                logprobs=_extract_logprobs(candidate),
            )
        )

    if not choices:
        choices.append(OpenAICompletionChoice(text="", finish_reason="stop", index=index_offset))

    return OpenAICompletion(
        id=completion_id,
        choices=choices,
        created=int(time.time()),
        model=model,
    )


def convert_completion_prompt_to_contents(prompt: str) -> list[dict[str, Any]]:
    """Wrap a plain-text prompt as a Gemini ``contents`` list for ``generate_content``."""
    return [{"role": "user", "parts": [{"text": prompt}]}]


def convert_gemini_response_to_openai_completion(
    response: genai_types.GenerateContentResponse,
    model: str,
    prompt: str,
) -> OpenAICompletion:
    """Map a google-genai ``GenerateContentResponse`` to ``OpenAICompletion`` (text completions)."""
    candidates = getattr(response, "candidates", None) or []

    choices: list[OpenAICompletionChoice] = []
    if candidates:
        for i, candidate in enumerate(candidates):
            text = (
                getattr(candidate.content.parts[0], "text", "") if candidate.content and candidate.content.parts else ""
            )
            finish_reason = _resolve_finish_reason(candidate.finish_reason, has_tool_calls=False)
            choices.append(
                OpenAICompletionChoice(
                    text=text,
                    finish_reason=finish_reason,
                    index=i,
                    logprobs=_extract_logprobs(candidate),
                )
            )
    else:
        choices.append(OpenAICompletionChoice(text="", finish_reason="content_filter", index=0))

    return OpenAICompletion(
        id=generate_completion_id(),
        choices=choices,
        created=int(time.time()),
        model=model,
    )
