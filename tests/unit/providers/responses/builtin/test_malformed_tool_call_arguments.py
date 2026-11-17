# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for #6819: malformed tool-call arguments crashed the whole
response (JSONDecodeError/AttributeError escaping out of execute_tool_call) instead of
being reported back to the model as a retriable tool error.
"""

from ogx_api.openai_responses import OpenAIResponseInputToolWebSearch
from ogx_api.responses.models import CreateResponseRequest
from ogx_api.tools import ToolDef
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


def _register_web_search_tool(openai_responses_impl):
    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )


async def test_malformed_json_arguments_dont_crash_and_are_fed_back_to_the_model(
    openai_responses_impl, mock_inference_api
):
    """A truncated JSON string (the exact repro from the issue) must not raise
    JSONDecodeError out of execute_tool_call -- the model should see the parse error as
    a tool message and get a chance to retry, exactly like any other tool failure.
    """
    _register_web_search_tool(openai_responses_impl)
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_malformed_json_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What's the weather in Boston?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )

    # The response completed -- it did not fail with server_error.
    assert result.status == "completed"
    # The malformed call never reached the tool runtime.
    openai_responses_impl.tool_runtime_api.invoke_tool.assert_not_called()

    # The second inference call's messages must include a tool message reporting the
    # parse failure, keyed to the original malformed tool call, so the model could see
    # and act on it.
    second_call_messages = mock_inference_api.openai_chat_completion.call_args_list[1].args[0].messages
    tool_messages = [m for m in second_call_messages if getattr(m, "role", None) == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "tool_call_malformed_123"
    assert "Failed to parse tool call arguments as a JSON object" in tool_messages[0].content

    # The failed web_search_call output item is still reported, marked failed.
    web_search_output = next((o for o in result.output if o.type == "web_search_call"), None)
    assert web_search_output is not None
    assert web_search_output.status == "failed"


async def test_non_object_json_arguments_dont_crash_and_are_fed_back_to_the_model(
    openai_responses_impl, mock_inference_api
):
    """Valid JSON that isn't an object (e.g. a bare list) must not raise AttributeError
    from a downstream tool_kwargs.get(...) call -- same treatment as unparsable JSON.
    """
    _register_web_search_tool(openai_responses_impl)
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_non_object_args_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What's the weather in Boston?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )

    assert result.status == "completed"
    openai_responses_impl.tool_runtime_api.invoke_tool.assert_not_called()

    second_call_messages = mock_inference_api.openai_chat_completion.call_args_list[1].args[0].messages
    tool_messages = [m for m in second_call_messages if getattr(m, "role", None) == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "tool_call_non_object_args_123"
    assert "Failed to parse tool call arguments as a JSON object" in tool_messages[0].content
    assert "list" in tool_messages[0].content


async def test_persistent_malformed_arguments_end_incomplete_not_failed(openai_responses_impl, mock_inference_api):
    """If the model keeps producing bad arguments past max_infer_iters, the response
    must end as incomplete (the existing, generic iteration-cap outcome any persistent
    tool failure already gets), never as a hard server_error failure.
    """
    _register_web_search_tool(openai_responses_impl)
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_malformed_json_completion.yaml"),
        fake_stream("tool_call_malformed_json_completion.yaml"),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What's the weather in Boston?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
            max_infer_iters=2,
        )
    )

    assert result.status == "incomplete"
    assert result.incomplete_details is not None
    assert result.incomplete_details.reason == "max_iterations_exceeded"
    openai_responses_impl.tool_runtime_api.invoke_tool.assert_not_called()
