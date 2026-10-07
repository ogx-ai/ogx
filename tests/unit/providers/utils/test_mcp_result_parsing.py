# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from mcp import types as mcp_types

from ogx.providers.utils.tools.mcp import _parse_mcp_result
from ogx_api.common.content_types import TextContentItem
from ogx_api.tools.models import ToolInvocationResult


def _call_tool_result(**wire: object) -> mcp_types.CallToolResult:
    # Build from the wire format so the test does not depend on SDK field aliases.
    return mcp_types.CallToolResult.model_validate({"content": [], **wire})


def _texts(result: ToolInvocationResult) -> list[str]:
    return [item.text for item in result.content if isinstance(item, TextContentItem)]


def test_structured_only_result_is_serialized_for_the_model():
    result = _parse_mcp_result(_call_tool_result(structuredContent={"temperature": 21.5, "unit": "C"}))

    assert result.error_code == 0
    assert _texts(result) == [json.dumps({"temperature": 21.5, "unit": "C"})]
    assert result.metadata == {"structured_content": {"temperature": 21.5, "unit": "C"}}


def test_structured_content_does_not_duplicate_text_content():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[{"type": "text", "text": "21.5 C"}],
            structuredContent={"temperature": 21.5, "unit": "C"},
        )
    )

    assert _texts(result) == ["21.5 C"]
    assert result.metadata == {"structured_content": {"temperature": 21.5, "unit": "C"}}


def test_resource_link_and_audio_blocks_do_not_fail_the_call():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[
                {
                    "type": "resource_link",
                    "uri": "file:///reports/q3.pdf",
                    "name": "q3.pdf",
                    "description": "Quarterly report",
                },
                {"type": "audio", "data": "UklGRg==", "mimeType": "audio/wav"},
                {"type": "text", "text": "done"},
            ]
        )
    )

    link, audio, text = _texts(result)
    assert "file:///reports/q3.pdf" in link and "Quarterly report" in link
    assert "audio/wav" in audio
    assert text == "done"
    assert result.error_code == 0


def test_embedded_resources_contribute_text_or_a_description():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[
                {
                    "type": "resource",
                    "resource": {"uri": "file:///notes.txt", "mimeType": "text/plain", "text": "hello"},
                },
                {
                    "type": "resource",
                    "resource": {"uri": "file:///photo.png", "mimeType": "image/png", "blob": "iVBORw0="},
                },
            ]
        )
    )

    text, binary = _texts(result)
    assert text == "hello"
    assert "file:///photo.png" in binary and "image/png" in binary


def test_tool_error_keeps_its_message():
    result = _parse_mcp_result(_call_tool_result(content=[{"type": "text", "text": "division by zero"}], isError=True))

    assert result.error_code == 1
    assert result.error_message == "division by zero"
    assert _texts(result) == ["division by zero"]
