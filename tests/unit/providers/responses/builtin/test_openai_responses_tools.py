# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

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
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    WebSearchToolTypes,
)
from ogx_api.tools import ListToolDefsResponse, ToolDef, ToolInvocationResult
from ogx_api.vector_io import (
    VectorStoreContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResponsePage,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


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
            input=input_text,
            model=model,
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolWebSearch(
                    name=tool_name,
                )
            ],
        )

        # Verify
        first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
        first_params = first_call.args[0]
        assert first_params.messages[0].content == "What is the capital of Ireland?"
        assert first_params.tools is not None
        assert first_params.temperature == 0.1

        second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
        second_params = second_call.args[0]
        assert second_params.messages[-1].content == "Dublin"
        assert second_params.temperature == 0.1

        openai_responses_impl.tool_groups_api.get_tool.assert_called_once_with("web_search")
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
        input="What is 2+2?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        store=True,
        tools=[
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ],
    )
    args = mock_responses_store.upsert_response_object.call_args
    data = args.kwargs["response_object"].model_dump()
    data["input"] = [input_item.model_dump() for input_item in args.kwargs["input"]]
    data["messages"] = [msg.model_dump() for msg in args.kwargs["messages"]]
    stored = _OpenAIResponseObjectWithInputAndMessages(**data)
    mock_responses_store.get_response_object.return_value = stored

    res2 = await openai_responses_impl.create_openai_response(
        previous_response_id=res1.id,
        input="Now what is 3+3?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        store=True,
        tools=[
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ],
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
        input="Test connector resolution",
        model="meta-llama/Llama-3.1-8B-Instruct",
        store=True,
        tools=[
            OpenAIResponseInputToolMCP(server_label="my-label", connector_id="my-mcp-connector"),
        ],
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
        input=input_text,
        model=model,
        stream=True,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_weather",
                description="Get current temperature for a given location.",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
            )
        ],
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
        input=input_text,
        model=model,
        stream=True,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_weather",
                description="Get current temperature for a given location.",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
            )
        ],
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
        input=input_text,
        model=model,
        stream=False,
        tools=[
            OpenAIResponseInputToolFunction(
                type="function",
                name="get_weather",
                description="Get weather information",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
                # strict is None by default
            )
        ],
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
        input=input_text,
        model=model,
        stream=False,
        tools=[
            OpenAIResponseInputToolFunction(
                type="function",
                name="get_weather",
                description="Get weather information",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
                strict=True,  # Explicitly set to True
            )
        ],
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
        input=input_text,
        model=model,
        stream=False,
        tools=[
            OpenAIResponseInputToolFunction(
                type="function",
                name="get_weather",
                description="Get weather information",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
                strict=False,  # Explicitly set to False
            )
        ],
    )

    # Verify the call was made
    assert mock_inference_api.openai_chat_completion.call_count == 1
    params = mock_inference_api.openai_chat_completion.call_args[0][0]

    # Verify 'strict' field IS present and set to False
    tool_function = params.tools[0]["function"]
    assert "strict" in tool_function, "strict field should be included when explicitly set to False"
    assert tool_function["strict"] is False, "strict field should be False"
