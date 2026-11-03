# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from ogx.core.datatypes import VectorStoresConfig
from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor
from ogx.providers.utils.responses.responses_store import (
    _OpenAIResponseObjectWithInputAndMessages,
)
from ogx_api import (
    GetConnectorRequest,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolNamespace,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    OpenAIResponseObjectStreamResponseContentPartDone,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseReasoning,
    WebSearchToolTypes,
)
from ogx_api.responses.models import CreateResponseRequest
from ogx_api.tools import ListToolDefsResponse, ToolDef, ToolInvocationResult
from ogx_api.vector_io import (
    VectorStoreContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_namespace_tool_flattened_to_function_tools_for_backend(openai_responses_impl, mock_inference_api):
    """Namespaced tools are accepted and flattened to function tools for the backend."""
    input_text = "Create a worktree"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=False,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_weather",
                    description="Get current temperature for a given location.",
                    parameters={"type": "object", "properties": {"location": {"type": "string"}}},
                ),
                OpenAIResponseInputToolNamespace(
                    name="codex_worktrees",
                    description="Tools in the codex_worktrees namespace.",
                    tools=[
                        OpenAIResponseInputToolFunction(
                            name="create_worktree",
                            description="Create a managed Git worktree.",
                            parameters={"type": "object", "properties": {"ref": {"type": "string"}}},
                        ),
                        OpenAIResponseInputToolFunction(
                            name="list_worktrees",
                            parameters={"type": "object", "properties": {}},
                        ),
                    ],
                ),
            ],
        )
    )

    assert mock_inference_api.openai_chat_completion.call_count == 1
    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    assert params.tools is not None
    names = [tool["function"]["name"] for tool in params.tools]
    # Namespaced functions are qualified at the chat layer so same-named
    # tools from different namespaces stay distinguishable.
    assert names == ["get_weather", "codex_worktrees__create_worktree", "codex_worktrees__list_worktrees"]
    assert all(tool["type"] == "function" for tool in params.tools)


async def test_namespace_tool_call_surfaces_as_function_call(openai_responses_impl, mock_inference_api):
    """A model call to a namespaced function is matched by bare name and surfaced to the client."""
    input_text = "Create a worktree"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_toolcall():
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
                                # The chat layer sees the qualified name.
                                function=ChoiceDeltaToolCallFunction(
                                    name="codex_worktrees__create_worktree", arguments="{}"
                                ),
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

    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            tools=[
                OpenAIResponseInputToolNamespace(
                    name="codex_worktrees",
                    description="Worktree tools.",
                    tools=[
                        OpenAIResponseInputToolFunction(
                            name="create_worktree",
                            parameters={"type": "object", "properties": {}},
                        ),
                    ],
                ),
            ],
        )
    )

    chunks = [chunk async for chunk in result]
    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert len(completed_chunk.response.output) == 1
    call = completed_chunk.response.output[0]
    assert call.type == "function_call"
    assert call.name == "create_worktree"
    # The namespace is echoed so clients (codex) can route the call to the handler that owns the group.
    assert call.namespace == "codex_worktrees"


def test_merge_namespace_tool_description():
    from ogx.providers.inline.responses.builtin.responses.utils import merge_namespace_tool_description

    # Both present: namespace context first, member description intact.
    assert merge_namespace_tool_description("Worktree tools.", "Create a worktree.") == (
        "Worktree tools. Create a worktree."
    )
    # Member without a description: just the namespace context.
    assert merge_namespace_tool_description("Worktree tools.", None) == "Worktree tools."
    assert merge_namespace_tool_description("Worktree tools.", "") == "Worktree tools."
    # No length limit: chat function descriptions are unbounded, so the merge
    # preserves both parts in full.
    long_ns = "x" * 5000
    assert merge_namespace_tool_description(long_ns, "Do it.") == f"{long_ns} Do it."
    assert merge_namespace_tool_description(long_ns, None) == long_ns


def test_split_qualified_tool_name():
    from ogx.providers.inline.responses.builtin.responses.utils import split_qualified_tool_name

    ns = lambda n, members: OpenAIResponseInputToolNamespace(  # noqa: E731
        name=n,
        description="d",
        tools=[OpenAIResponseInputToolFunction(name=m, parameters={"type": "object"}) for m in members],
    )

    # Qualified match returns the exact member, even when the namespace name
    # itself contains the delimiter.
    tools = [ns("a__b", ["x"])]
    assert split_qualified_tool_name("a__b__x", tools) == ("x", "a__b")
    # Two namespaces whose qualified forms collide are still exact per member.
    tools = [ns("a", ["b__x"]), ns("a__b", ["x"])]
    assert split_qualified_tool_name("a__b__x", tools) == ("b__x", "a")
    # Bare-name fallback: labeled when unambiguous.
    tools = [ns("solo", ["x"])]
    assert split_qualified_tool_name("x", tools) == ("x", "solo")
    # Bare name shared with a top-level function: no label.
    tools = [OpenAIResponseInputToolFunction(name="x", parameters={"type": "object"}), ns("solo", ["x"])]
    assert split_qualified_tool_name("x", tools) == ("x", None)
    # Bare name in two namespaces: no label.
    tools = [ns("one", ["x"]), ns("two", ["x"])]
    assert split_qualified_tool_name("x", tools) == ("x", None)
    # Unknown names pass through unchanged.
    assert split_qualified_tool_name("bogus", tools) == ("bogus", None)
    assert split_qualified_tool_name("one__bogus", tools) == ("one__bogus", None)


async def test_hallucinated_call_with_only_namespaced_tools_surfaces_to_client(
    openai_responses_impl, mock_inference_api
):
    """Namespaced tools count as client-side, so unrecognized names are surfaced, not looped."""
    input_text = "Do a thing that does not exist"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_bogus():
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
                                function=ChoiceDeltaToolCallFunction(name="totally_bogus", arguments="{}"),
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

    mock_inference_api.openai_chat_completion.return_value = fake_stream_bogus()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            tools=[
                OpenAIResponseInputToolNamespace(
                    name="solo",
                    description="Only group.",
                    tools=[
                        OpenAIResponseInputToolFunction(
                            name="real_tool",
                            parameters={"type": "object", "properties": {}},
                        ),
                    ],
                ),
            ],
        )
    )

    chunks = [chunk async for chunk in result]
    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert completed_chunk.response.status == "completed"
    call = completed_chunk.response.output[0]
    assert call.type == "function_call"
    assert call.name == "totally_bogus"
    assert call.namespace is None
    # No retry loop: a single inference call.
    assert mock_inference_api.openai_chat_completion.call_count == 1


async def test_tool_choice_prefers_top_level_function_over_namespaced(openai_responses_impl, mock_inference_api):
    input_text = "Pick one"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=False,
            tools=[
                OpenAIResponseInputToolFunction(name="x", parameters={"type": "object", "properties": {}}),
                OpenAIResponseInputToolNamespace(
                    name="ns",
                    description="Group.",
                    tools=[
                        OpenAIResponseInputToolFunction(name="x", parameters={"type": "object", "properties": {}}),
                    ],
                ),
            ],
            tool_choice={"type": "function", "name": "x"},
        )
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    assert params.tool_choice == {"type": "function", "function": {"name": "x"}}


async def test_qualified_name_collision_rejected(openai_responses_impl, mock_inference_api):
    """A top-level function named like a qualified namespaced tool is a 400."""
    from ogx_api import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="x",
                model="meta-llama/Llama-3.1-8B-Instruct",
                stream=False,
                tools=[
                    OpenAIResponseInputToolFunction(name="ns__x", parameters={"type": "object", "properties": {}}),
                    OpenAIResponseInputToolNamespace(
                        name="ns",
                        description="Group.",
                        tools=[
                            OpenAIResponseInputToolFunction(name="x", parameters={"type": "object", "properties": {}}),
                        ],
                    ),
                ],
            )
        )


async def test_duplicate_member_names_in_one_namespace_allowed(openai_responses_impl, mock_inference_api):
    """Same member name twice in a namespace is allowed (OpenAI overload pattern)."""
    input_text = "Call the amount variant"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_toolcall():
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
                                function=ChoiceDeltaToolCallFunction(
                                    name="alpha__dup_overload", arguments='{"amount":5}'
                                ),
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

    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            reasoning=OpenAIResponseReasoning(effort="none"),
            tools=[
                OpenAIResponseInputToolNamespace(
                    name="alpha",
                    description="Probe namespace",
                    tools=[
                        OpenAIResponseInputToolFunction(
                            name="dup_overload",
                            description="No-arg variant",
                            parameters={"type": "object", "properties": {}},
                        ),
                        OpenAIResponseInputToolFunction(
                            name="dup_overload",
                            description="Amount variant",
                            parameters={
                                "type": "object",
                                "properties": {"amount": {"type": "integer"}},
                                "required": ["amount"],
                            },
                        ),
                    ],
                ),
            ],
        )
    )

    chunks = [chunk async for chunk in result]

    # Both overloads reach the chat layer under the same qualified name,
    # distinguished by their schemas.
    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    qualified = [t for t in params.tools if t["function"]["name"] == "alpha__dup_overload"]
    assert len(qualified) == 2

    call = chunks[-1].response.output[0]
    assert call.type == "function_call"
    assert call.name == "dup_overload"
    assert call.namespace == "alpha"
    assert call.arguments == '{"amount":5}'


async def test_namespace_description_merged_into_chat_tools(openai_responses_impl, mock_inference_api):
    input_text = "Create a worktree"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=False,
            tools=[
                OpenAIResponseInputToolNamespace(
                    name="codex_worktrees",
                    description="Worktree tools.",
                    tools=[
                        OpenAIResponseInputToolFunction(
                            name="create_worktree",
                            description="Create a managed worktree.",
                            parameters={"type": "object", "properties": {}},
                        ),
                        OpenAIResponseInputToolFunction(
                            name="list_worktrees",
                            parameters={"type": "object", "properties": {}},
                        ),
                    ],
                ),
            ],
        )
    )

    params = mock_inference_api.openai_chat_completion.call_args[0][0]
    by_name = {tool["function"]["name"]: tool["function"] for tool in params.tools}
    assert by_name["codex_worktrees__create_worktree"]["description"] == ("Worktree tools. Create a managed worktree.")
    assert by_name["codex_worktrees__list_worktrees"]["description"] == "Worktree tools."


async def test_namespaced_tool_call_identity_survives_name_collisions(openai_responses_impl, mock_inference_api):
    """Qualified chat-layer names let the exact tool be identified on collisions."""
    input_text = "Do the thing"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    def dup_namespace(name: str, description: str) -> OpenAIResponseInputToolNamespace:
        return OpenAIResponseInputToolNamespace(
            name=name,
            description=description,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="dup_func",
                    parameters={"type": "object", "properties": {}},
                ),
            ],
        )

    # Same bare name in two namespaces, plus a top-level function with the
    # same bare name: the qualified chat-layer name identifies the exact tool.
    tools = [
        OpenAIResponseInputToolFunction(
            name="dup_func",
            description="Top-level variant.",
            parameters={"type": "object", "properties": {}},
        ),
        dup_namespace("alpha", "First group."),
        dup_namespace("beta", "Second group."),
    ]

    cases = [
        # (chat-layer name the model calls, expected (name, namespace))
        ("dup_func", ("dup_func", None)),
        ("alpha__dup_func", ("dup_func", "alpha")),
        ("beta__dup_func", ("dup_func", "beta")),
    ]

    for called_name, (expected_name, expected_namespace) in cases:

        async def fake_stream_toolcall(called_name=called_name):
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
                                    function=ChoiceDeltaToolCallFunction(name=called_name, arguments="{}"),
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

        mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=input_text,
                model=model,
                stream=True,
                tools=tools,
            )
        )

        chunks = [chunk async for chunk in result]
        completed_chunk = chunks[-1]
        assert completed_chunk.type == "response.completed"
        call = completed_chunk.response.output[0]
        assert call.type == "function_call"
        assert call.name == expected_name
        assert call.namespace == expected_namespace


async def test_create_openai_response_with_string_input_with_tools(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a simple string input and tools."""
    # Setup
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The query to search for"}},
            "required": ["query"],
        },
    )

    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Dublin",
    )

    # Execute
    for tool_name in WebSearchToolTypes:
        # Reset mock states as we loop through each tool type
        mock_inference_api.openai_chat_completion.side_effect = [
            fake_stream("tool_call_completion.yaml"),
            fake_stream(),
        ]
        openai_responses_impl.tool_groups_api.get_tool.reset_mock()
        openai_responses_impl.tool_runtime_api.invoke_tool.reset_mock()
        openai_responses_impl.responses_store.upsert_response_object.reset_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=input_text,
                model=model,
                temperature=0.1,
                tools=[
                    OpenAIResponseInputToolWebSearch(
                        type=tool_name,
                    )
                ],
            )
        )

        # Verify
        first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
        first_params = first_call.args[0]
        assert first_params.messages[0].content == "What is the capital of Ireland?"
        assert first_params.tools is not None
        assert first_params.temperature == 0.1

        second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
        second_params = second_call.args[0]
        # web_search results are wrapped in untrusted-content delimiters (#6263)
        # before being fed back to the model -- the raw content is no longer
        # passed through verbatim.
        assert "Dublin" in second_params.messages[-1].content
        assert second_params.temperature == 0.1

        openai_responses_impl.tool_groups_api.get_tool.assert_called_with("web_search")
        openai_responses_impl.tool_runtime_api.invoke_tool.assert_called_once_with(
            tool_name="web_search",
            kwargs={"query": "What is the capital of Ireland?"},
        )

        openai_responses_impl.responses_store.upsert_response_object.assert_called()

        # Check that we got the content from our mocked tool execution result
        assert len(result.output) >= 1
        assert isinstance(result.output[1], OpenAIResponseMessage)
        assert result.output[1].content[0].text == "Dublin"
        assert result.output[1].content[0].annotations == []


async def test_create_openai_response_with_tool_call_type_none(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a tool call response that has a type of None."""
    # Setup
    input_text = "How hot it is in San Francisco today?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_toolcall():
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
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments="{}"),
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

    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_weather",
                    description="Get current temperature for a given location.",
                    parameters={
                        "location": "string",
                    },
                )
            ],
        )
    )

    # Check that we got the content from our mocked tool execution result
    chunks = [chunk async for chunk in result]

    # Verify event types
    # Should have: response.created, response.in_progress, output_item.added,
    # function_call_arguments.delta, function_call_arguments.done, output_item.done, response.completed
    assert len(chunks) == 7

    event_types = [chunk.type for chunk in chunks]
    assert event_types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    # Verify inference API was called correctly (after iterating over result)
    first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
    first_params = first_call.args[0]
    assert first_params.messages[0].content == input_text
    assert first_params.tools is not None
    assert first_params.temperature == 0.1

    # Check response.created event (should have empty output)
    assert len(chunks[0].response.output) == 0

    # Check response.completed event (should have the tool call)
    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert len(completed_chunk.response.output) == 1
    assert completed_chunk.response.output[0].type == "function_call"
    assert completed_chunk.response.output[0].name == "get_weather"


async def test_create_openai_response_with_tool_call_function_arguments_none(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with tool calls that omit arguments."""

    input_text = "What is the time right now?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_toolcall():
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
                                function=ChoiceDeltaToolCallFunction(name="get_current_time", arguments=None),
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

    def assert_common_expectations(chunks) -> None:
        first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
        first_params = first_call.args[0]
        assert first_params.messages[0].content == input_text
        assert first_params.tools is not None
        assert first_params.temperature == 0.1
        assert len(chunks[0].response.output) == 0
        completed_chunk = chunks[-1]
        assert completed_chunk.type == "response.completed"
        assert len(completed_chunk.response.output) == 1
        assert completed_chunk.response.output[0].type == "function_call"
        assert completed_chunk.response.output[0].name == "get_current_time"
        assert completed_chunk.response.output[0].arguments == "{}"

    # Function does not accept arguments
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()
    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_current_time", description="Get current time for system's timezone", parameters={}
                )
            ],
        )
    )
    chunks = [chunk async for chunk in result]
    assert [chunk.type for chunk in chunks] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert_common_expectations(chunks)

    # Function accepts optional arguments
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()
    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_current_time",
                    description="Get current time for system's timezone",
                    parameters={"timezone": "string"},
                )
            ],
        )
    )
    chunks = [chunk async for chunk in result]
    assert [chunk.type for chunk in chunks] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert_common_expectations(chunks)

    # Function accepts optional arguments with additional optional fields
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()
    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_current_time",
                    description="Get current time for system's timezone",
                    parameters={"timezone": "string", "location": "string"},
                )
            ],
        )
    )
    chunks = [chunk async for chunk in result]
    assert [chunk.type for chunk in chunks] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert_common_expectations(chunks)
    mock_inference_api.openai_chat_completion.return_value = fake_stream_toolcall()


@patch("ogx.providers.inline.responses.builtin.responses.streaming.list_mcp_tools")
async def test_reuse_mcp_tool_list(
    mock_list_mcp_tools, openai_responses_impl, mock_responses_store, mock_inference_api
):
    """Test that mcp_list_tools can be reused where appropriate."""

    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    mock_list_mcp_tools.return_value = ListToolDefsResponse(
        data=[ToolDef(name="test_tool", description="a test tool", input_schema={}, output_schema={})]
    )

    res1 = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is 2+2?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            store=True,
            tools=[
                OpenAIResponseInputToolFunction(name="fake", parameters=None),
                OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
            ],
        )
    )
    args = mock_responses_store.upsert_response_object.call_args
    data = args.kwargs["response_object"].model_dump()
    data["input"] = [input_item.model_dump() for input_item in args.kwargs["input"]]
    data["messages"] = [msg.model_dump() for msg in args.kwargs["messages"]]
    stored = _OpenAIResponseObjectWithInputAndMessages(**data)
    mock_responses_store.get_response_object.return_value = stored

    res2 = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            previous_response_id=res1.id,
            input="Now what is 3+3?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            store=True,
            tools=[
                OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
            ],
        )
    )
    assert len(mock_inference_api.openai_chat_completion.call_args_list) == 2
    second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
    second_params = second_call.args[0]
    tools_seen = second_params.tools
    assert len(tools_seen) == 1
    assert tools_seen[0]["function"]["name"] == "test_tool"
    assert tools_seen[0]["function"]["description"] == "a test tool"

    assert mock_list_mcp_tools.call_count == 1
    listings = [obj for obj in res2.output if obj.type == "mcp_list_tools"]
    assert len(listings) == 1
    assert listings[0].server_label == "alabel"
    assert len(listings[0].tools) == 1
    assert listings[0].tools[0].name == "test_tool"


@patch("ogx.providers.inline.responses.builtin.responses.streaming.list_mcp_tools")
async def test_mcp_tool_connector_id_resolved_to_server_url(
    mock_list_mcp_tools, openai_responses_impl, mock_responses_store, mock_inference_api, mock_connectors_api
):
    """Test that connector_id is resolved to server_url when using MCP tools."""
    from ogx_api import Connector, ConnectorType

    # Setup mock connector that will be returned when resolving connector_id
    mock_connector = Connector(
        connector_id="my-mcp-connector",
        connector_type=ConnectorType.MCP,
        url="http://resolved-mcp-server:8080/mcp",
        server_label="Resolved MCP Server",
    )
    mock_connectors_api.get_connector.return_value = mock_connector

    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    mock_list_mcp_tools.return_value = ListToolDefsResponse(
        data=[ToolDef(name="resolved_tool", description="a resolved tool", input_schema={}, output_schema={})]
    )

    # Create a response using connector_id instead of server_url
    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="Test connector resolution",
            model="meta-llama/Llama-3.1-8B-Instruct",
            store=True,
            tools=[
                OpenAIResponseInputToolMCP(server_label="my-label", connector_id="my-mcp-connector"),
            ],
        )
    )

    # Verify the connector_id was resolved via the connectors API
    mock_connectors_api.get_connector.assert_called_once_with(GetConnectorRequest(connector_id="my-mcp-connector"))

    # Verify list_mcp_tools was called with the resolved URL
    mock_list_mcp_tools.assert_called_once()
    call_kwargs = mock_list_mcp_tools.call_args.kwargs
    assert call_kwargs["endpoint"] == "http://resolved-mcp-server:8080/mcp"

    # Verify the response contains the resolved tools
    listings = [obj for obj in result.output if obj.type == "mcp_list_tools"]
    assert len(listings) == 1
    assert listings[0].server_label == "my-label"
    assert len(listings[0].tools) == 1
    assert listings[0].tools[0].name == "resolved_tool"


async def test_file_search_uses_default_search_mode_from_config(mock_vector_io_api):
    """Test that file_search tool executor passes default_search_mode from VectorStoresConfig."""
    from ogx.core.datatypes import ChunkRetrievalParams

    query = "What is machine learning?"
    vector_store_id = "test_vector_store"

    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=[query],
        has_more=False,
        data=[],
    )

    # Test with hybrid mode configured
    hybrid_config = VectorStoresConfig(
        chunk_retrieval_params=ChunkRetrievalParams(default_search_mode="hybrid"),
    )
    tool_executor = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=hybrid_config,
        mcp_session_manager=None,
    )

    file_search_tool = OpenAIResponseInputToolFileSearch(vector_store_ids=[vector_store_id])
    await tool_executor._execute_file_search_via_vector_store(
        query=query,
        response_file_search_tool=file_search_tool,
    )

    # Verify search_mode="hybrid" was passed in the request
    call_kwargs = mock_vector_io_api.openai_search_vector_store.call_args
    request = call_kwargs.kwargs["request"]
    assert request.search_mode == "hybrid", f"Expected search_mode='hybrid', got '{request.search_mode}'"

    # Test with default config (should use "vector")
    mock_vector_io_api.openai_search_vector_store.reset_mock()
    default_config = VectorStoresConfig()
    tool_executor_default = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=default_config,
        mcp_session_manager=None,
    )

    await tool_executor_default._execute_file_search_via_vector_store(
        query=query,
        response_file_search_tool=file_search_tool,
    )

    call_kwargs = mock_vector_io_api.openai_search_vector_store.call_args
    request = call_kwargs.kwargs["request"]
    assert request.search_mode == "vector", f"Expected search_mode='vector', got '{request.search_mode}'"


async def test_file_search_results_include_chunk_metadata_attributes(mock_vector_io_api):
    """Test that file_search tool executor preserves chunk metadata attributes."""
    query = "What is machine learning?"
    vector_store_id = "test_vector_store"

    # Mock vector_io to return search results with custom attributes
    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=[query],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="doc-123",
                filename="ml-intro.md",
                content=[VectorStoreContent(type="text", text="Machine learning is a subset of AI")],
                score=0.95,
                attributes={
                    "document_id": "ml-intro",
                    "source_url": "https://example.com/ml-guide",
                    "title": "Introduction to ML",
                    "author": "John Doe",
                    "year": "2024",
                },
            ),
            VectorStoreSearchResponse(
                file_id="doc-456",
                filename="dl-basics.md",
                content=[VectorStoreContent(type="text", text="Deep learning uses neural networks")],
                score=0.85,
                attributes={
                    "document_id": "dl-basics",
                    "source_url": "https://example.com/dl-guide",
                    "title": "Deep Learning Basics",
                    "category": "tutorial",
                },
            ),
        ],
    )

    # Create tool executor with mock vector_io
    tool_executor = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=VectorStoresConfig(),
        mcp_session_manager=None,
    )

    # Execute the file search
    file_search_tool = OpenAIResponseInputToolFileSearch(vector_store_ids=[vector_store_id])
    result = await tool_executor._execute_file_search_via_vector_store(
        query=query,
        response_file_search_tool=file_search_tool,
    )

    mock_vector_io_api.openai_search_vector_store.assert_called_once()

    # Verify the result metadata includes chunk attributes
    assert result.metadata is not None
    assert "attributes" in result.metadata
    attributes = result.metadata["attributes"]
    assert len(attributes) == 2

    # Verify first result has all expected attributes
    attrs1 = attributes[0]
    assert attrs1["document_id"] == "ml-intro"
    assert attrs1["source_url"] == "https://example.com/ml-guide"
    assert attrs1["title"] == "Introduction to ML"
    assert attrs1["author"] == "John Doe"
    assert attrs1["year"] == "2024"

    # Verify second result has its attributes
    attrs2 = attributes[1]
    assert attrs2["document_id"] == "dl-basics"
    assert attrs2["source_url"] == "https://example.com/dl-guide"
    assert attrs2["title"] == "Deep Learning Basics"
    assert attrs2["category"] == "tutorial"

    # Verify scores and document_ids are also present
    assert result.metadata["scores"] == [0.95, 0.85]
    assert result.metadata["document_ids"] == ["doc-123", "doc-456"]
    assert result.metadata["chunks"] == [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
    ]


async def test_file_search_works_without_explicit_vector_stores_config(mock_vector_io_api):
    """Test that file_search works when vector_stores_config is not passed to ToolExecutor."""
    query = "What is machine learning?"
    vector_store_id = "test_vector_store"

    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=[query],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="doc-123",
                filename="ml-intro.md",
                content=[VectorStoreContent(type="text", text="Machine learning is a subset of AI")],
                score=0.95,
                attributes={},
            ),
        ],
    )

    tool_executor = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        mcp_session_manager=None,
    )

    file_search_tool = OpenAIResponseInputToolFileSearch(vector_store_ids=[vector_store_id])
    result = await tool_executor._execute_file_search_via_vector_store(
        query=query,
        response_file_search_tool=file_search_tool,
    )

    mock_vector_io_api.openai_search_vector_store.assert_called_once()
    assert result.content is not None


async def test_tool_call_arguments_arrive_in_subsequent_delta(openai_responses_impl, mock_inference_api):
    """Test that tool call arguments are correctly accumulated when the model streams
    arguments=None in the first delta and actual arguments in a subsequent delta.

    This is the streaming pattern used by vllm with llama3_json tool call parser:
    - Delta 1: function name + index, arguments=None
    - Delta 2: actual arguments JSON

    Regression test for bug where arguments were initialized to "{}" causing
    concatenation like '{}{"location": "..."}' which is invalid JSON.
    """
    input_text = "What is the weather in San Francisco?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_vllm_style():
        # Delta 1: function name arrives, arguments is None (vllm streaming behavior)
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
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=None),
                                type="function",
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )
        # Delta 2: actual arguments arrive in subsequent chunk
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                function=ChoiceDeltaToolCallFunction(
                                    name=None, arguments='{"location": "San Francisco"}'
                                ),
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

    mock_inference_api.openai_chat_completion.return_value = fake_stream_vllm_style()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_weather",
                    description="Get current temperature for a given location.",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            ],
        )
    )
    chunks = [chunk async for chunk in result]

    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert len(completed_chunk.response.output) == 1
    assert completed_chunk.response.output[0].type == "function_call"
    assert completed_chunk.response.output[0].name == "get_weather"
    # Arguments must be valid JSON — not '{}{"location": "..."}' which was the bug
    import json

    parsed = json.loads(completed_chunk.response.output[0].arguments)
    assert parsed == {"location": "San Francisco"}


async def test_tool_call_arguments_split_across_multiple_deltas(openai_responses_impl, mock_inference_api):
    """Test that tool call arguments are correctly accumulated when streamed
    across more than two deltas (name, partial args, remaining args).
    """
    input_text = "What is the weather in Boston?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_split_args():
        # Delta 1: function name, no arguments
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
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=None),
                                type="function",
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )
        # Delta 2: first part of arguments
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                function=ChoiceDeltaToolCallFunction(name=None, arguments='{"location":'),
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
        # Delta 3: remaining arguments
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                function=ChoiceDeltaToolCallFunction(name=None, arguments=' "Boston"}'),
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

    mock_inference_api.openai_chat_completion.return_value = fake_stream_split_args()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_weather",
                    description="Get current temperature for a given location.",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            ],
        )
    )
    chunks = [chunk async for chunk in result]

    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert completed_chunk.response.output[0].type == "function_call"
    assert completed_chunk.response.output[0].name == "get_weather"

    import json

    parsed = json.loads(completed_chunk.response.output[0].arguments)
    assert parsed == {"location": "Boston"}


async def test_tool_call_no_parameters_still_returns_empty_json(openai_responses_impl, mock_inference_api):
    """Test that a no-parameter function with arguments=None across all deltas
    still produces valid '{}' arguments at the end of streaming.
    """
    input_text = "What time is it?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream_no_args():
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
                                function=ChoiceDeltaToolCallFunction(name="get_current_time", arguments=None),
                                type="function",
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream_no_args()

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=True,
            tools=[
                OpenAIResponseInputToolFunction(
                    name="get_current_time",
                    description="Get the current time",
                    parameters={},
                )
            ],
        )
    )
    chunks = [chunk async for chunk in result]

    completed_chunk = chunks[-1]
    assert completed_chunk.type == "response.completed"
    assert completed_chunk.response.output[0].type == "function_call"
    assert completed_chunk.response.output[0].name == "get_current_time"
    # No-parameter functions should produce valid empty JSON
    import json

    parsed = json.loads(completed_chunk.response.output[0].arguments)
    assert parsed == {}


async def test_function_tool_strict_field_excluded_when_none(openai_responses_impl, mock_inference_api):
    """Test that function tool 'strict' field is excluded when None (fix for #4617)."""
    input_text = "What is the weather?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Mock inference response
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute with function tool that has strict=None (default)
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=False,
            tools=[
                OpenAIResponseInputToolFunction(
                    type="function",
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                    # strict is None by default
                )
            ],
        )
    )

    # Verify the call was made
    assert mock_inference_api.openai_chat_completion.call_count == 1
    params = mock_inference_api.openai_chat_completion.call_args[0][0]

    # Verify tools were passed
    assert params.tools is not None
    assert len(params.tools) == 1

    # Critical: verify 'strict' field is NOT present when it's None
    # This prevents "strict: null" from being sent to OpenAI API
    tool_function = params.tools[0]["function"]
    assert "strict" not in tool_function, (
        "strict field should be excluded when None to avoid OpenAI API validation error"
    )

    # Verify other fields are present
    assert tool_function["name"] == "get_weather"
    assert tool_function["description"] == "Get weather information"
    assert tool_function["parameters"] is not None


async def test_function_tool_strict_field_included_when_set(openai_responses_impl, mock_inference_api):
    """Test that function tool 'strict' field is included when explicitly set."""
    input_text = "What is the weather?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Mock inference response
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute with function tool that has strict=True
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=False,
            tools=[
                OpenAIResponseInputToolFunction(
                    type="function",
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                    strict=True,  # Explicitly set to True
                )
            ],
        )
    )

    # Verify the call was made
    assert mock_inference_api.openai_chat_completion.call_count == 1
    params = mock_inference_api.openai_chat_completion.call_args[0][0]

    # Verify tools were passed
    assert params.tools is not None
    assert len(params.tools) == 1

    # Verify 'strict' field IS present when explicitly set
    tool_function = params.tools[0]["function"]
    assert "strict" in tool_function, "strict field should be included when explicitly set"
    assert tool_function["strict"] is True, "strict field should have the correct value"

    # Verify other fields are present
    assert tool_function["name"] == "get_weather"
    assert tool_function["description"] == "Get weather information"


async def test_function_tool_strict_false_included(openai_responses_impl, mock_inference_api):
    """Test that function tool 'strict' field is included when set to False."""
    input_text = "What is the weather?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Mock inference response
    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute with function tool that has strict=False
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text,
            model=model,
            stream=False,
            tools=[
                OpenAIResponseInputToolFunction(
                    type="function",
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                    strict=False,  # Explicitly set to False
                )
            ],
        )
    )

    # Verify the call was made
    assert mock_inference_api.openai_chat_completion.call_count == 1
    params = mock_inference_api.openai_chat_completion.call_args[0][0]

    # Verify 'strict' field IS present and set to False
    tool_function = params.tools[0]["function"]
    assert "strict" in tool_function, "strict field should be included when explicitly set to False"
    assert tool_function["strict"] is False, "strict field should be False"


async def test_file_search_citation_populated_when_model_cites_inline(
    openai_responses_impl, mock_inference_api, mock_vector_io_api
):
    """Regression test for annotations never being populated for file_search results.

    Round 1: the model calls file_search. Round 2: the model answers, citing the
    retrieved file with the instructed `<|file-id|>` marker. The final response's
    output_text annotations should contain a matching file_citation.
    """
    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["What is global warming?"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file-abc123",
                filename="climate.pdf",
                score=0.9,
                content=[VectorStoreContent(type="text", text="Greenhouse gases trap heat.")],
            )
        ],
    )
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("file_search_tool_call_completion.yaml"),
        fake_stream("file_search_final_answer_completion.yaml"),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is global warming in two lines?",
            model="ollama/llama3.2:3b",
            tools=[OpenAIResponseInputToolFileSearch(type="file_search", vector_store_ids=["vs_1"])],
        )
    )

    messages = [item for item in result.output if isinstance(item, OpenAIResponseMessage)]
    assert len(messages) == 1
    text_content = messages[0].content[0]
    assert text_content.text == "Global warming is caused by greenhouse gases."
    assert len(text_content.annotations) == 1
    assert text_content.annotations[0].file_id == "file-abc123"
    assert text_content.annotations[0].filename == "climate.pdf"


async def test_file_search_citation_falls_back_when_model_does_not_cite(
    openai_responses_impl, mock_inference_api, mock_vector_io_api
):
    """When a weak/local model ignores the citation-marker instruction entirely, the
    response should still attribute the answer to the single highest-scoring retrieved
    file rather than returning empty annotations, and rather than citing every file
    retrieved this turn (which would overstate how many sources actually back the answer).
    """
    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["What is global warming?"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file-abc123",
                filename="climate.pdf",
                score=0.9,
                content=[VectorStoreContent(type="text", text="Greenhouse gases trap heat.")],
            ),
            VectorStoreSearchResponse(
                file_id="file-def456",
                filename="unrelated.pdf",
                score=0.3,
                content=[VectorStoreContent(type="text", text="Some other, less relevant chunk.")],
            ),
        ],
    )

    async def uncited_final_answer():
        yield ChatCompletionChunk(
            id="chat-completion-458",
            created=1234567892,
            model="ollama/llama3.2:3b",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Global warming is caused by greenhouse gases trapping heat.",
                    ),
                    finish_reason="stop",
                )
            ],
        )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("file_search_tool_call_completion.yaml"),
        uncited_final_answer(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is global warming in two lines?",
            model="ollama/llama3.2:3b",
            tools=[OpenAIResponseInputToolFileSearch(type="file_search", vector_store_ids=["vs_1"])],
        )
    )

    messages = [item for item in result.output if isinstance(item, OpenAIResponseMessage)]
    assert len(messages) == 1
    text_content = messages[0].content[0]
    assert text_content.text == "Global warming is caused by greenhouse gases trapping heat."
    assert len(text_content.annotations) == 1
    assert text_content.annotations[0].file_id == "file-abc123"
    assert text_content.annotations[0].filename == "climate.pdf"


async def test_file_search_citation_falls_back_when_model_does_not_cite_streaming(
    openai_responses_impl, mock_inference_api, mock_vector_io_api
):
    """Same scenario as test_file_search_citation_falls_back_when_model_does_not_cite, but
    driven through the actual `stream=True` SSE path rather than the aggregated response.

    The two non-streaming citation tests above only ever inspect the final aggregated
    OpenAIResponseObject; they never consume the raw event stream, so they never exercise
    (i.e. assert against) the content_part.done / output_item.done events built directly in
    streaming.py's per-chunk processing loop. This drives the same fixtures through
    `request.stream=True` and asserts those streamed events themselves carry the
    single-highest-scoring-file fallback citation, not one annotation per retrieved file.
    Two files are retrieved (with different scores) so that "cite only the best one" is
    actually distinguishable from the old "cite every retrieved file" behavior.
    """
    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["What is global warming?"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file-abc123",
                filename="climate.pdf",
                score=0.9,
                content=[VectorStoreContent(type="text", text="Greenhouse gases trap heat.")],
            ),
            VectorStoreSearchResponse(
                file_id="file-def456",
                filename="unrelated.pdf",
                score=0.3,
                content=[VectorStoreContent(type="text", text="Some other, less relevant chunk.")],
            ),
        ],
    )

    async def uncited_final_answer():
        yield ChatCompletionChunk(
            id="chat-completion-458",
            created=1234567892,
            model="ollama/llama3.2:3b",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Global warming is caused by greenhouse gases trapping heat.",
                    ),
                    finish_reason="stop",
                )
            ],
        )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("file_search_tool_call_completion.yaml"),
        uncited_final_answer(),
    ]

    stream = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is global warming in two lines?",
            model="ollama/llama3.2:3b",
            tools=[OpenAIResponseInputToolFileSearch(type="file_search", vector_store_ids=["vs_1"])],
            stream=True,
        )
    )

    content_part_done_events = []
    output_item_done_message_events = []
    async for event in stream:
        if isinstance(event, OpenAIResponseObjectStreamResponseContentPartDone):
            content_part_done_events.append(event)
        elif isinstance(event, OpenAIResponseObjectStreamResponseOutputItemDone) and isinstance(
            event.item, OpenAIResponseMessage
        ):
            output_item_done_message_events.append(event)

    # The final answer round is the only one that streams actual text content.
    final_content_part_events = [e for e in content_part_done_events if e.part.text]
    assert len(final_content_part_events) == 1
    part = final_content_part_events[0].part
    assert part.text == "Global warming is caused by greenhouse gases trapping heat."
    assert len(part.annotations) == 1
    assert part.annotations[0].file_id == "file-abc123"
    assert part.annotations[0].filename == "climate.pdf"

    final_output_item_events = [e for e in output_item_done_message_events if e.item.content]
    assert len(final_output_item_events) == 1
    item_content = final_output_item_events[0].item.content[0]
    assert item_content.text == "Global warming is caused by greenhouse gases trapping heat."
    assert len(item_content.annotations) == 1
    assert item_content.annotations[0].file_id == "file-abc123"
    assert item_content.annotations[0].filename == "climate.pdf"


async def test_streaming_deltas_stay_consistent_with_final_text_when_marker_split_across_chunks(
    openai_responses_impl, mock_inference_api, mock_vector_io_api
):
    """Regression test: text delta events must stay consistent with the cleaned text in
    content_part.done, even when a citation marker is split across multiple raw provider
    chunks (a real possibility with live token-by-token streaming). Concatenating every
    delta's text must reproduce exactly the cleaned final text, and no individual delta may
    ever leak a fragment of the raw marker syntax.
    """
    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["What is global warming?"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file-abc123",
                filename="climate.pdf",
                score=0.9,
                content=[VectorStoreContent(type="text", text="Greenhouse gases trap heat.")],
            )
        ],
    )

    async def split_marker_final_answer():
        # "<|file-abc123|>" is deliberately split across three separate chunk deltas.
        chunks_content = ["Global warming is caused by greenhouse gases ", "<|file-abc", "123|>", "."]
        for i, delta_text in enumerate(chunks_content):
            yield ChatCompletionChunk(
                id="chat-completion-999",
                created=1234567893,
                model="ollama/llama3.2:3b",
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(role="assistant", content=delta_text),
                        finish_reason="stop" if i == len(chunks_content) - 1 else None,
                    )
                ],
            )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("file_search_tool_call_completion.yaml"),
        split_marker_final_answer(),
    ]

    stream = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is global warming in two lines?",
            model="ollama/llama3.2:3b",
            tools=[OpenAIResponseInputToolFileSearch(type="file_search", vector_store_ids=["vs_1"])],
            stream=True,
        )
    )

    delta_texts = []
    content_part_done_text = None
    async for event in stream:
        if isinstance(event, OpenAIResponseObjectStreamResponseOutputTextDelta):
            delta_texts.append(event.delta)
            assert "<|file-" not in event.delta, "raw marker syntax must never leak into a delta"
            assert "|>" not in event.delta, "raw marker syntax must never leak into a delta"
        elif isinstance(event, OpenAIResponseObjectStreamResponseContentPartDone) and event.part.text:
            content_part_done_text = event.part.text

    assert content_part_done_text == "Global warming is caused by greenhouse gases."
    assert "".join(delta_texts) == content_part_done_text
