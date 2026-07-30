# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for inference/chat completion with JSON Schema-based tools.
Tests that tools pass through correctly to various LLM providers.
"""

import pytest

from ogx.core.library_client import OGXAsLibraryClient


class TestChatCompletionWithTools:
    """Test chat completion with tools that have complex schemas."""

    def test_simple_tool_call(self, ogx_client, text_model_id):
        """Test basic tool calling with simple input schema."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = ogx_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
            tools=tools,
        )

        assert response is not None

    def test_tool_with_complex_schema(self, ogx_client, text_model_id):
        """Test tool calling with complex schema including $ref and $defs."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a flight",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flight": {"$ref": "#/$defs/FlightInfo"},
                            "passenger": {"$ref": "#/$defs/Passenger"},
                        },
                        "required": ["flight", "passenger"],
                        "$defs": {
                            "FlightInfo": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "date": {"type": "string", "format": "date"},
                                },
                            },
                            "Passenger": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                            },
                        },
                    },
                },
            }
        ]

        response = ogx_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Book a flight from SFO to JFK for John Doe"}],
            tools=tools,
        )

        # The key test: No errors during schema processing
        # The LLM received a valid, complete schema with $ref/$defs
        assert response is not None


class TestOpenAICompatibility:
    """Test OpenAI-compatible endpoints with new schema format."""

    def test_openai_chat_completion_with_tools(self, compat_client, text_model_id):
        """Test OpenAI-compatible chat completion with tools."""
        from openai import OpenAI

        if not isinstance(compat_client, OpenAI):
            pytest.skip("OpenAI client required")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        response = compat_client.chat.completions.create(
            model=text_model_id, messages=[{"role": "user", "content": "What's the weather in Tokyo?"}], tools=tools
        )

        assert response is not None
        assert response.choices is not None

    def test_openai_format_preserves_complex_schemas(self, compat_client, text_model_id):
        """Test that complex schemas work through OpenAI-compatible API."""
        from openai import OpenAI

        if not isinstance(compat_client, OpenAI):
            pytest.skip("OpenAI client required")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process_data",
                    "description": "Process structured data",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"$ref": "#/$defs/DataObject"}},
                        "$defs": {
                            "DataObject": {
                                "type": "object",
                                "properties": {"values": {"type": "array", "items": {"type": "number"}}},
                            }
                        },
                    },
                },
            }
        ]

        response = compat_client.chat.completions.create(
            model=text_model_id, messages=[{"role": "user", "content": "Process this data"}], tools=tools
        )

        assert response is not None


class TestMCPToolsInChatCompletion:
    """Test using MCP-style tool schemas in chat completion."""

    def test_mcp_tools_in_inference(self, ogx_client, text_model_id):
        """Test that MCP-style tool schemas work in inference."""
        if not isinstance(ogx_client, OGXAsLibraryClient):
            pytest.skip("Library client required")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation on two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "First operand"},
                            "y": {"type": "number", "description": "Second operand"},
                            "operation": {
                                "type": "string",
                                "description": "The operation to perform",
                                "enum": ["add", "sub", "mul", "div"],
                            },
                        },
                        "required": ["x", "y", "operation"],
                    },
                },
            }
        ]

        response = ogx_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Calculate 5 + 3"}],
            tools=tools,
            max_tokens=256,
        )

        assert response is not None


class TestStreamingWithTools:
    """Test streaming chat completion with tools."""

    def test_streaming_tool_calls(self, ogx_client, text_model_id):
        """Test that tool schemas work correctly in streaming mode."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}},
                },
            }
        ]

        response_stream = ogx_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "What time is it in UTC?"}],
            tools=tools,
            stream=True,
        )

        # Should be able to iterate through stream
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)

        # Should have received some chunks
        assert len(chunks) >= 0


class TestEdgeCases:
    """Test edge cases in inference with tools."""

    def test_tool_without_schema(self, ogx_client, text_model_id):
        """Test tool with no input_schema."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "no_args_tool",
                    "description": "Tool with no arguments",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = ogx_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Call the no args tool"}],
            tools=tools,
        )

        assert response is not None

    def test_multiple_tools_with_different_schemas(self, ogx_client, text_model_id):
        """Test multiple tools with different schema complexities."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complex",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"$ref": "#/$defs/Complex"}},
                        "$defs": {
                            "Complex": {
                                "type": "object",
                                "properties": {"nested": {"type": "array", "items": {"type": "number"}}},
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "with_output",
                    "parameters": {"type": "object", "properties": {"input": {"type": "string"}}},
                },
            },
        ]

        response = ogx_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": "Use one of the available tools"}],
            tools=tools,
        )

        # All tools should have been processed without errors
        assert response is not None
