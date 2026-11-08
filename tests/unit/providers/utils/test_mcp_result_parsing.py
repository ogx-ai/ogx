# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from mcp import types as mcp_types

from ogx.providers.utils.tools.mcp import _parse_mcp_result
from ogx_api.common.content_types import ImageContentItem, TextContentItem
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


def test_unknown_block_type_is_described_instead_of_dropped():
    class FutureBlock:
        type = "hologram"

    class Result:
        content = [FutureBlock()]
        structured_content = None
        is_error = False

    result = _parse_mcp_result(Result())

    assert result.error_code == 0
    assert _texts(result) == ["[unsupported MCP content block: FutureBlock]"]


def test_image_blocks_pass_through_as_image_items():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[
                {"type": "image", "data": "iVBORw0=", "mimeType": "image/png"},
                {"type": "text", "text": "a chart"},
            ]
        )
    )

    image, text = result.content
    assert isinstance(image, ImageContentItem)
    assert image.image.data == "iVBORw0="
    assert isinstance(text, TextContentItem)
    assert text.text == "a chart"
    assert result.error_code == 0


def test_error_without_text_content_has_no_error_message():
    image_only = _parse_mcp_result(
        _call_tool_result(content=[{"type": "image", "data": "iVBORw0=", "mimeType": "image/png"}], isError=True)
    )
    assert image_only.error_code == 1
    assert image_only.error_message is None

    empty = _parse_mcp_result(_call_tool_result(isError=True))
    assert empty.error_code == 1
    assert empty.error_message is None


def test_error_message_joins_all_text_blocks():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}],
            isError=True,
        )
    )

    assert result.error_code == 1
    assert result.error_message == "line one\nline two"


def test_error_keeps_its_message_and_structured_content():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[{"type": "text", "text": "boom"}],
            isError=True,
            structuredContent={"failed_step": "parse"},
        )
    )

    assert result.error_code == 1
    assert result.error_message == "boom"
    assert result.metadata == {"structured_content": {"failed_step": "parse"}}


def test_resource_link_without_description_has_no_trailing_colon():
    result = _parse_mcp_result(
        _call_tool_result(content=[{"type": "resource_link", "uri": "file:///reports/q3.pdf", "name": "q3.pdf"}])
    )

    assert _texts(result) == ["[resource link file:///reports/q3.pdf (q3.pdf)]"]


def test_embedded_blob_resource_without_mime_type_uses_a_fallback():
    result = _parse_mcp_result(
        _call_tool_result(content=[{"type": "resource", "resource": {"uri": "file:///photo.png", "blob": "iVBORw0="}}])
    )

    assert _texts(result) == ["[binary resource file:///photo.png, unknown type]"]


def test_non_object_structured_content_is_serialized_for_the_model():
    result = _parse_mcp_result(_call_tool_result(structuredContent=[1, 2, 3]))

    assert result.error_code == 0
    assert _texts(result) == [json.dumps([1, 2, 3])]
    assert result.metadata == {"structured_content": [1, 2, 3]}


def test_structured_content_is_not_duplicated_when_images_are_present():
    result = _parse_mcp_result(
        _call_tool_result(
            content=[{"type": "image", "data": "iVBORw0=", "mimeType": "image/png"}],
            structuredContent={"temperature": 21.5},
        )
    )

    assert len(result.content) == 1
    assert isinstance(result.content[0], ImageContentItem)
    assert result.metadata == {"structured_content": {"temperature": 21.5}}
