# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the "namespace" tool type (#6938): groups function tools under a
named wrapper, e.g. clients such as codex send

    {"type": "namespace", "name": "codex_worktrees", "description": "...",
     "tools": [{"type": "function", "name": "list_worktrees", "parameters": {...}}]}

The model calls namespaced functions by their bare name ("list_worktrees"), and
the resulting function_call output item carries a namespace field back so the
caller can route the call to the handler that owns the group.
"""

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from ogx_api.openai_responses import (
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolNamespace,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


def _namespace_tool(name: str = "codex_worktrees") -> OpenAIResponseInputToolNamespace:
    return OpenAIResponseInputToolNamespace(
        name=name,
        description=f"Tools in the {name} namespace.",
        tools=[
            OpenAIResponseInputToolFunction(
                name="list_worktrees",
                description="List git worktrees",
                parameters={"type": "object", "properties": {}},
            ),
            OpenAIResponseInputToolFunction(
                name="create_worktree",
                description="Create a git worktree",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ],
    )


async def _fake_stream_namespaced_call(model: str):
    yield ChatCompletionChunk(
        id="123",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="tc_123",
                            function=ChoiceDeltaToolCallFunction(name="list_worktrees", arguments="{}"),
                            type=None,
                        )
                    ]
                ),
            ),
        ],
        created=1,
        model=model,
        object="chat.completion.chunk",
    )


async def test_namespace_tool_is_accepted_and_flattened_into_chat_tools(openai_responses_impl, mock_inference_api):
    """The request must not be rejected (the reported 400 union_tag_invalid), and
    the nested function tools must be flattened into the chat-completion request
    by their bare name -- the model calls them without any namespace prefix.
    """
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="list the worktrees",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[_namespace_tool()],
        )
    )

    params = mock_inference_api.openai_chat_completion.call_args_list[0].args[0]
    assert params.tools is not None
    sent_names = {tool["function"]["name"] for tool in params.tools}
    assert sent_names == {"list_worktrees", "create_worktree"}
    assert all(tool["type"] == "function" for tool in params.tools)


async def test_namespaced_function_call_carries_namespace_back(openai_responses_impl, mock_inference_api):
    """When the model calls a namespaced function by its bare name, the resulting
    function_call output item must carry the owning namespace back so the client
    can route the call to the right handler.
    """
    model = "meta-llama/Llama-3.1-8B-Instruct"
    mock_inference_api.openai_chat_completion.return_value = _fake_stream_namespaced_call(model)

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="list the worktrees",
            model=model,
            stream=True,
            tools=[_namespace_tool()],
        )
    )

    chunks = [chunk async for chunk in result]
    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert len(completed_chunk.response.output) == 1
    function_call = completed_chunk.response.output[0]
    assert function_call.type == "function_call"
    assert function_call.name == "list_worktrees"
    assert function_call.namespace == "codex_worktrees"


async def test_plain_function_tool_has_no_namespace(openai_responses_impl, mock_inference_api):
    """A function tool declared directly (not inside a namespace) must still have
    namespace=None on its output item -- confirming the new field doesn't leak
    across unrelated tool calls.
    """
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_plain_call():
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc_456",
                                function=ChoiceDeltaToolCallFunction(name="get_current_time", arguments="{}"),
                                type=None,
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_plain_call()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="what time is it?",
            model=model,
            stream=True,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_current_time", description="Get the current time", parameters={}
                )
            ],
        )
    )

    chunks = [chunk async for chunk in result]
    completed_chunk = chunks[-1]
    function_call = completed_chunk.response.output[0]
    assert function_call.type == "function_call"
    assert function_call.namespace is None


async def test_two_namespaces_with_same_bare_name_route_independently(openai_responses_impl, mock_inference_api):
    """Two namespaces can each expose a tool with the same bare name (the model
    only ever sees bare names); the server tracks the most recently processed
    namespace for that name since namespaces are processed in request order.
    """
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_shared_name_call():
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc_789",
                                function=ChoiceDeltaToolCallFunction(name="list_worktrees", arguments="{}"),
                                type=None,
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_shared_name_call()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="list the worktrees",
            model=model,
            stream=True,
            tools=[_namespace_tool("codex_worktrees"), _namespace_tool("other_worktrees")],
        )
    )

    chunks = [chunk async for chunk in result]
    completed_chunk = chunks[-1]
    function_call = completed_chunk.response.output[0]
    assert function_call.namespace == "other_worktrees"
