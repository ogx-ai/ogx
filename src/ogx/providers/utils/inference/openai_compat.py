# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import (
    Any,
)

from pydantic import BaseModel

from ogx.log import get_logger

logger = get_logger(name=__name__, category="providers::utils")


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
