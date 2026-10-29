# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from typing import (
    Any,
)

from pydantic import BaseModel

from ogx.log import get_logger

logger = get_logger(name=__name__, category="providers::utils")


def _classify_json_value_kind(leading_char: str) -> str:
    """Best-effort classification of a JSON value from its first non-whitespace
    character, for diagnostics only -- it does not confirm the rest is valid JSON.
    """
    if leading_char == "[":
        return "array"
    if leading_char == '"':
        return "string"
    if leading_char in "-0123456789":
        return "number"
    if leading_char in "tf":
        return "boolean"
    if leading_char == "n":
        return "null"
    return "invalid"


def normalize_tool_call_arguments(arguments: str | None, *, tool_name: str = "") -> dict[str, Any]:
    """Parse a tool call's raw JSON arguments into a dict, substituting {} for
    anything else.

    Both Anthropic's ``tool_use.input`` and Google's function-call ``args`` are
    always objects on their respective real APIs; a model can return any JSON value
    as OpenAI-style tool-call arguments (a list, string, number, etc.), or invalid or
    truncated JSON, none of which the client expects here. Empty/``None`` arguments
    (the common no-arguments case) silently become {} with no warning; a non-empty
    value that isn't an object logs one, without the arguments themselves (#5851).
    """
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning(
            "Tool call arguments were not valid JSON; substituting {}",
            tool=tool_name,
        )
        return {}
    if isinstance(parsed, dict):
        return parsed
    logger.warning(
        "Tool call arguments were not a JSON object; substituting {}",
        tool=tool_name,
        value_kind=type(parsed).__name__,
    )
    return {}


class ToolCallArgumentsStreamFilter:
    """Per-tool-call state machine that makes a streamed tool call's arguments end up
    shaped like :func:`normalize_tool_call_arguments` would shape the same arguments
    non-streaming, without buffering the whole argument string (which would delay
    every streaming tool call, valid or not, until it finishes).

    Decides from the first non-whitespace character of the arguments: ``{`` means the
    value can only be a JSON object, so every chunk streams through immediately with
    no added delay; anything else means it cannot be, so a single ``"{}"`` delta is
    sent instead and every further chunk for that tool call is dropped. Only leading
    whitespace is ever held back pending that decision.

    Known limitation: a value that starts with ``{`` but turns out to be invalid or
    truncated JSON is not caught -- by the time that's known, the real chunks already
    streamed to the client and cannot be taken back. Catching that case too would mean
    buffering each tool call's entire argument string, defeating the point of this
    class.
    """

    def __init__(self, tool_name: str = "") -> None:
        self._tool_name = tool_name
        self._buffer = ""
        # None = not yet decided; True = confirmed object, streaming through as-is;
        # False = confirmed non-object, fallback already sent, further chunks dropped.
        self._decided: bool | None = None

    def feed(self, chunk: str) -> str | None:
        """Process one raw argument chunk. Returns the text to forward as this tool
        call's next delta, or None if nothing should be forwarded yet.
        """
        if self._decided is True:
            return chunk
        if self._decided is False:
            return None

        self._buffer += chunk
        stripped = self._buffer.lstrip()
        if not stripped:
            return None  # still all whitespace so far; keep holding it back

        if stripped[0] == "{":
            self._decided = True
            flushed, self._buffer = self._buffer, ""
            return flushed

        self._decided = False
        self._buffer = ""
        logger.warning(
            "Streamed tool call arguments were not a JSON object; substituting {}",
            tool=self._tool_name,
            value_kind=_classify_json_value_kind(stripped[0]),
        )
        return "{}"

    def finalize(self) -> str | None:
        """Call once after the tool call's argument stream has ended. Returns a
        trailing "{}" delta to forward if the tool call never received any
        non-whitespace argument content (e.g. truly empty arguments, matching
        normalize_tool_call_arguments' silent, non-warning {} for that case), else
        None.
        """
        if self._decided is None:
            self._decided = False
            return "{}"
        return None


def convert_tooldef_to_openai_tool(
    tool_name: str,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert tool parameters to an OpenAI API-compatible dictionary.

    Args:
        tool_name: Tool name as string
        description: Optional tool description
        input_schema: Optional JSON Schema for tool parameters

    Returns:
        OpenAI-compatible tool dictionary:
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {<JSON Schema>},
            },
        }

    NOTE: OpenAI does not support output_schema, so it is not included.
    """
    out: dict[str, Any] = {
        "type": "function",
        "function": {},
    }
    function: dict[str, Any] = out["function"]

    function["name"] = tool_name

    if description:
        function["description"] = description

    if input_schema:
        function["parameters"] = input_schema

    return out


async def prepare_openai_completion_params(**params):
    """Prepare keyword arguments for OpenAI-compatible completion calls.

    Args:
        **params: parameter key-value pairs, where Pydantic models are serialized

    Returns:
        Dictionary of non-None parameters with models converted to dicts
    """

    async def _prepare_value(value: Any) -> Any:
        new_value = value
        if isinstance(value, list):
            new_value = [await _prepare_value(v) for v in value]
        elif isinstance(value, dict):
            new_value = {k: await _prepare_value(v) for k, v in value.items()}
        elif isinstance(value, BaseModel):
            new_value = value.model_dump(exclude_none=True)
        return new_value

    completion_params = {}
    for k, v in params.items():
        if v is not None:
            completion_params[k] = await _prepare_value(v)
    return completion_params


def get_stream_options_for_telemetry(
    stream_options: dict[str, Any] | None,
    is_streaming: bool,
    supports_stream_options: bool = True,
) -> dict[str, Any] | None:
    """
    Inject stream_options when streaming and telemetry is active.

    Active telemetry takes precedence over caller preference to ensure
    complete and consistent observability metrics.

    Args:
        stream_options: Existing stream options from the request
        is_streaming: Whether this is a streaming request
        supports_stream_options: Whether the provider supports stream_options parameter

    Returns:
        Updated stream_options with include_usage=True if conditions are met, otherwise original options
    """
    if not is_streaming:
        return stream_options

    if not supports_stream_options:
        return stream_options

    from opentelemetry import trace

    span = trace.get_current_span()
    if not span or not span.is_recording():
        return stream_options

    if stream_options is None:
        return {"include_usage": True}

    return {**stream_options, "include_usage": True}
