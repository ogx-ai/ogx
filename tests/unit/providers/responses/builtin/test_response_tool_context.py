# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.providers.inline.responses.builtin.responses.streaming import (
    _process_tool_choice,
)
from ogx.providers.inline.responses.builtin.responses.types import ToolContext
from ogx_api import (
    MCPListToolsTool,
    OpenAIChatCompletionToolChoiceAllowedTools,
    OpenAIChatCompletionToolChoiceCustomTool,
    OpenAIChatCompletionToolChoiceFunctionTool,
    OpenAIResponseInputToolChoiceAllowedTools,
    OpenAIResponseInputToolChoiceCustomTool,
    OpenAIResponseInputToolChoiceFileSearch,
    OpenAIResponseInputToolChoiceFunctionTool,
    OpenAIResponseInputToolChoiceMCPTool,
    OpenAIResponseInputToolChoiceMode,
    OpenAIResponseInputToolChoiceWebSearch,
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseToolMCP,
)


class TestToolContext:
    def test_no_tools(self):
        tools = []
        context = ToolContext(tools)
        previous_response = OpenAIResponseObject(
            created_at=1234, id="test", model="mymodel", output=[], status="", store=True
        )
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 0
        assert len(context.previous_tools) == 0
        assert len(context.previous_tool_listings) == 0

    def test_no_previous_tools(self):
        tools = [
            OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
            OpenAIResponseInputToolMCP(server_label="label", server_url="url"),
        ]
        context = ToolContext(tools)
        previous_response = OpenAIResponseObject(
            created_at=1234, id="test", model="mymodel", output=[], status="", store=True
        )
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 2
        assert len(context.previous_tools) == 0
        assert len(context.previous_tool_listings) == 0

    def test_reusable_server(self):
        tools = [
            OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test", server_label="alabel", tools=[MCPListToolsTool(name="test_tool", input_schema={})]
            )
        ]
        previous_response = OpenAIResponseObject(
            created_at=1234, id="test", model="fake", output=output, status="", store=True
        )
        previous_response.tools = [
            OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
            OpenAIResponseToolMCP(server_label="alabel"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 1
        assert context.tools_to_process[0].type == "file_search"
        assert len(context.previous_tools) == 1
        assert context.previous_tools["test_tool"].server_label == "alabel"
        assert context.previous_tools["test_tool"].server_url == "aurl"
        assert len(context.previous_tool_listings) == 1
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "alabel"

    def test_multiple_reusable_servers(self):
        tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test1", server_label="alabel", tools=[MCPListToolsTool(name="test_tool", input_schema={})]
            ),
            OpenAIResponseOutputMessageMCPListTools(
                id="test2",
                server_label="anotherlabel",
                tools=[MCPListToolsTool(name="some_other_tool", input_schema={})],
            ),
        ]
        previous_response = OpenAIResponseObject(
            created_at=1234, id="test", model="fake", output=output, status="", store=True
        )
        previous_response.tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 2
        assert context.tools_to_process[0].type == "function"
        assert context.tools_to_process[1].type == "web_search"
        assert len(context.previous_tools) == 2
        assert context.previous_tools["test_tool"].server_label == "alabel"
        assert context.previous_tools["test_tool"].server_url == "aurl"
        assert context.previous_tools["some_other_tool"].server_label == "anotherlabel"
        assert context.previous_tools["some_other_tool"].server_url == "anotherurl"
        assert len(context.previous_tool_listings) == 2
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "alabel"
        assert len(context.previous_tool_listings[1].tools) == 1
        assert context.previous_tool_listings[1].server_label == "anotherlabel"

    def test_multiple_servers_only_one_reusable(self):
        tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test2",
                server_label="anotherlabel",
                tools=[MCPListToolsTool(name="some_other_tool", input_schema={})],
            )
        ]
        previous_response = OpenAIResponseObject(
            created_at=1234, id="test", model="fake", output=output, status="", store=True
        )
        previous_response.tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 3
        assert context.tools_to_process[0].type == "function"
        assert context.tools_to_process[1].type == "web_search"
        assert context.tools_to_process[2].type == "mcp"
        assert len(context.previous_tools) == 1
        assert context.previous_tools["some_other_tool"].server_label == "anotherlabel"
        assert context.previous_tools["some_other_tool"].server_url == "anotherurl"
        assert len(context.previous_tool_listings) == 1
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "anotherlabel"

    def test_mismatched_allowed_tools(self):
        tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseInputToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseInputToolMCP(server_label="alabel", server_url="aurl", allowed_tools=["test_tool_2"]),
        ]
        context = ToolContext(tools)
        output = [
            OpenAIResponseOutputMessageMCPListTools(
                id="test1", server_label="alabel", tools=[MCPListToolsTool(name="test_tool_1", input_schema={})]
            ),
            OpenAIResponseOutputMessageMCPListTools(
                id="test2",
                server_label="anotherlabel",
                tools=[MCPListToolsTool(name="some_other_tool", input_schema={})],
            ),
        ]
        previous_response = OpenAIResponseObject(
            created_at=1234, id="test", model="fake", output=output, status="", store=True
        )
        previous_response.tools = [
            OpenAIResponseInputToolFunction(name="fake", parameters=None),
            OpenAIResponseToolMCP(server_label="anotherlabel", server_url="anotherurl"),
            OpenAIResponseInputToolWebSearch(type="web_search"),
            OpenAIResponseToolMCP(server_label="alabel", server_url="aurl"),
        ]
        context.recover_tools_from_previous_response(previous_response)

        assert len(context.tools_to_process) == 3
        assert context.tools_to_process[0].type == "function"
        assert context.tools_to_process[1].type == "web_search"
        assert context.tools_to_process[2].type == "mcp"
        assert len(context.previous_tools) == 1
        assert context.previous_tools["some_other_tool"].server_label == "anotherlabel"
        assert context.previous_tools["some_other_tool"].server_url == "anotherurl"
        assert len(context.previous_tool_listings) == 1
        assert len(context.previous_tool_listings[0].tools) == 1
        assert context.previous_tool_listings[0].server_label == "anotherlabel"


class TestProcessToolChoice:
    """Comprehensive test suite for _process_tool_choice function."""

    def setup_method(self):
        """Set up common test fixtures."""
        self.chat_tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "calculate"}},
            {"type": "function", "function": {"name": "file_search"}},
            {"type": "function", "function": {"name": "web_search"}},
        ]
        self.server_label_to_tools = {
            "mcp_server_1": ["mcp_tool_1", "mcp_tool_2"],
            "mcp_server_2": ["mcp_tool_3"],
        }

    async def test_mode_auto(self):
        """Test auto mode - should return 'auto' string."""
        tool_choice = OpenAIResponseInputToolChoiceMode.auto
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)
        assert result == "auto"

    async def test_mode_none(self):
        """Test none mode - should return 'none' string."""
        tool_choice = OpenAIResponseInputToolChoiceMode.none
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)
        assert result == "none"

    async def test_mode_required_with_tools(self):
        """Test required mode with available tools - should return AllowedTools with all function tools."""
        tool_choice = OpenAIResponseInputToolChoiceMode.required
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert result.allowed_tools.mode == "required"
        assert len(result.allowed_tools.tools) == 4
        tool_names = [tool["function"]["name"] for tool in result.allowed_tools.tools]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names
        assert "file_search" in tool_names
        assert "web_search" in tool_names

    async def test_mode_required_without_tools(self):
        """Test required mode without available tools - should return None."""
        tool_choice = OpenAIResponseInputToolChoiceMode.required
        result = await _process_tool_choice([], tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_allowed_tools_function(self):
        """Test allowed_tools with function tool types."""
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="required",
            tools=[
                {"type": "function", "name": "get_weather"},
                {"type": "function", "name": "calculate"},
            ],
        )
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert result.allowed_tools.mode == "required"
        assert len(result.allowed_tools.tools) == 2
        assert result.allowed_tools.tools[0]["function"]["name"] == "get_weather"
        assert result.allowed_tools.tools[1]["function"]["name"] == "calculate"

    async def test_allowed_tools_custom(self):
        """Test allowed_tools with custom tool types."""
        chat_tools = [{"type": "function", "function": {"name": "custom_tool_1"}}]
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="auto",
            tools=[{"type": "custom", "name": "custom_tool_1"}],
        )
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert result.allowed_tools.mode == "auto"
        assert len(result.allowed_tools.tools) == 1
        assert result.allowed_tools.tools[0]["type"] == "custom"
        assert result.allowed_tools.tools[0]["custom"]["name"] == "custom_tool_1"

    async def test_allowed_tools_file_search(self):
        """Test allowed_tools with file_search."""
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="required",
            tools=[{"type": "file_search"}],
        )
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert len(result.allowed_tools.tools) == 1
        assert result.allowed_tools.tools[0]["function"]["name"] == "file_search"

    async def test_allowed_tools_web_search(self):
        """Test allowed_tools with web_search."""
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="required",
            tools=[
                {"type": "web_search_preview_2025_03_11"},
                {"type": "web_search_2025_08_26"},
                {"type": "web_search_preview"},
            ],
        )
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert len(result.allowed_tools.tools) == 3
        assert result.allowed_tools.tools[0]["function"]["name"] == "web_search"
        assert result.allowed_tools.tools[0]["type"] == "function"
        assert result.allowed_tools.tools[1]["function"]["name"] == "web_search"
        assert result.allowed_tools.tools[1]["type"] == "function"
        assert result.allowed_tools.tools[2]["function"]["name"] == "web_search"
        assert result.allowed_tools.tools[2]["type"] == "function"

    async def test_allowed_tools_mcp_server_label(self):
        """Test allowed_tools with MCP server label (no specific tool name)."""
        chat_tools = [
            {"type": "function", "function": {"name": "mcp_tool_1"}},
            {"type": "function", "function": {"name": "mcp_tool_2"}},
        ]
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="required",
            tools=[{"type": "mcp", "server_label": "mcp_server_1"}],
        )
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert len(result.allowed_tools.tools) == 2
        tool_names = [tool["function"]["name"] for tool in result.allowed_tools.tools]
        assert "mcp_tool_1" in tool_names
        assert "mcp_tool_2" in tool_names

    async def test_allowed_tools_mixed_types(self):
        """Test allowed_tools with mixed tool types."""
        chat_tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "file_search"}},
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "mcp_tool_1"}},
        ]
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="auto",
            tools=[
                {"type": "function", "name": "get_weather"},
                {"type": "file_search"},
                {"type": "web_search"},
                {"type": "mcp", "server_label": "mcp_server_1"},
            ],
        )
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        # Should have: get_weather, file_search, web_search, mcp_tool_1, mcp_tool_2
        assert len(result.allowed_tools.tools) >= 3

    async def test_allowed_tools_invalid_type(self):
        """Test allowed_tools with an unsupported tool type - should skip it."""
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="required",
            tools=[
                {"type": "function", "name": "get_weather"},
                {"type": "unsupported_type", "name": "bad_tool"},
            ],
        )
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        # Should only include the valid function tool
        assert len(result.allowed_tools.tools) == 1
        assert result.allowed_tools.tools[0]["function"]["name"] == "get_weather"

    async def test_specific_custom_tool_valid(self):
        """Test specific custom tool choice when tool exists."""
        chat_tools = [{"type": "function", "function": {"name": "custom_tool"}}]
        tool_choice = OpenAIResponseInputToolChoiceCustomTool(name="custom_tool")
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceCustomTool)
        assert result.custom.name == "custom_tool"

    async def test_specific_custom_tool_invalid(self):
        """Test specific custom tool choice when tool doesn't exist - should return None."""
        tool_choice = OpenAIResponseInputToolChoiceCustomTool(name="nonexistent_tool")
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_specific_function_tool_valid(self):
        """Test specific function tool choice when tool exists."""
        tool_choice = OpenAIResponseInputToolChoiceFunctionTool(name="get_weather")
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceFunctionTool)
        assert result.function.name == "get_weather"

    async def test_specific_function_tool_invalid(self):
        """Test specific function tool choice when tool doesn't exist - should return None."""
        tool_choice = OpenAIResponseInputToolChoiceFunctionTool(name="nonexistent_function")
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_specific_file_search_valid(self):
        """Test file_search tool choice when available."""
        tool_choice = OpenAIResponseInputToolChoiceFileSearch()
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceFunctionTool)
        assert result.function.name == "file_search"

    async def test_specific_file_search_invalid(self):
        """Test file_search tool choice when not available - should return None."""
        chat_tools = [{"type": "function", "function": {"name": "get_weather"}}]
        tool_choice = OpenAIResponseInputToolChoiceFileSearch()
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_specific_web_search_valid(self):
        """Test web_search tool choice when available."""
        tool_choice = OpenAIResponseInputToolChoiceWebSearch()
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceFunctionTool)
        assert result.function.name == "web_search"

    async def test_specific_web_search_invalid(self):
        """Test web_search tool choice when not available - should return None."""
        chat_tools = [{"type": "function", "function": {"name": "get_weather"}}]
        tool_choice = OpenAIResponseInputToolChoiceWebSearch()
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_specific_mcp_tool_with_name(self):
        """Test MCP tool choice with specific tool name."""
        chat_tools = [{"type": "function", "function": {"name": "mcp_tool_1"}}]
        tool_choice = OpenAIResponseInputToolChoiceMCPTool(
            server_label="mcp_server_1",
            name="mcp_tool_1",
        )
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceFunctionTool)
        assert result.function.name == "mcp_tool_1"

    async def test_specific_mcp_tool_with_name_not_in_chat_tools(self):
        """Test MCP tool choice with specific tool name that doesn't exist in chat_tools."""
        chat_tools = [{"type": "function", "function": {"name": "other_tool"}}]
        tool_choice = OpenAIResponseInputToolChoiceMCPTool(
            server_label="mcp_server_1",
            name="mcp_tool_1",
        )
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_specific_mcp_tool_server_label_only(self):
        """Test MCP tool choice with only server label (no specific tool name)."""
        chat_tools = [
            {"type": "function", "function": {"name": "mcp_tool_1"}},
            {"type": "function", "function": {"name": "mcp_tool_2"}},
        ]
        tool_choice = OpenAIResponseInputToolChoiceMCPTool(server_label="mcp_server_1")
        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert result.allowed_tools.mode == "required"
        assert len(result.allowed_tools.tools) == 2
        tool_names = [tool["function"]["name"] for tool in result.allowed_tools.tools]
        assert "mcp_tool_1" in tool_names
        assert "mcp_tool_2" in tool_names

    async def test_specific_mcp_tool_unknown_server(self):
        """Test MCP tool choice with unknown server label."""
        tool_choice = OpenAIResponseInputToolChoiceMCPTool(
            server_label="unknown_server",
            name="some_tool",
        )
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)
        # Should return None because server not found
        assert result is None

    async def test_empty_chat_tools(self):
        """Test with empty chat_tools list."""
        tool_choice = OpenAIResponseInputToolChoiceFunctionTool(name="get_weather")
        result = await _process_tool_choice([], tool_choice, self.server_label_to_tools)
        assert result is None

    async def test_empty_server_label_to_tools(self):
        """Test with empty server_label_to_tools mapping."""
        tool_choice = OpenAIResponseInputToolChoiceMCPTool(server_label="mcp_server_1")
        result = await _process_tool_choice(self.chat_tools, tool_choice, {})
        # Should handle gracefully
        assert result is None or isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)

    async def test_allowed_tools_empty_list(self):
        """Test allowed_tools with empty tools list."""
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(mode="auto", tools=[])
        result = await _process_tool_choice(self.chat_tools, tool_choice, self.server_label_to_tools)

        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)
        assert len(result.allowed_tools.tools) == 0

    async def test_mcp_tool_multiple_servers(self):
        """Test MCP tool choice with multiple server labels."""
        chat_tools = [
            {"type": "function", "function": {"name": "mcp_tool_1"}},
            {"type": "function", "function": {"name": "mcp_tool_2"}},
            {"type": "function", "function": {"name": "mcp_tool_3"}},
        ]
        server_label_to_tools = {
            "server_a": ["mcp_tool_1"],
            "server_b": ["mcp_tool_2", "mcp_tool_3"],
        }

        # Test server_a
        tool_choice_a = OpenAIResponseInputToolChoiceMCPTool(server_label="server_a")
        result_a = await _process_tool_choice(chat_tools, tool_choice_a, server_label_to_tools)
        assert isinstance(result_a, OpenAIChatCompletionToolChoiceAllowedTools)
        assert len(result_a.allowed_tools.tools) == 1

        # Test server_b
        tool_choice_b = OpenAIResponseInputToolChoiceMCPTool(server_label="server_b")
        result_b = await _process_tool_choice(chat_tools, tool_choice_b, server_label_to_tools)
        assert isinstance(result_b, OpenAIChatCompletionToolChoiceAllowedTools)
        assert len(result_b.allowed_tools.tools) == 2

    async def test_allowed_tools_filters_effective_tools_correctly(self):
        """Test that allowed_tools properly filters effective_tools in streaming loop.

        This verifies the filtering logic used in streaming.py:
        1. Extract allowed tool names from processed_tool_choice
        2. Filter chat_tools to only include tools with names in allowed set
        3. Tools specified in allowed_tools but not in chat_tools are excluded
        """
        # Limited chat_tools - simulates what's actually available
        chat_tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "calculate"}},
        ]

        # Request includes a tool that doesn't exist in chat_tools
        tool_choice = OpenAIResponseInputToolChoiceAllowedTools(
            mode="required",
            tools=[
                {"type": "function", "name": "get_weather"},
                {"type": "function", "name": "nonexistent_tool"},  # Not in chat_tools
            ],
        )

        result = await _process_tool_choice(chat_tools, tool_choice, self.server_label_to_tools)
        assert isinstance(result, OpenAIChatCompletionToolChoiceAllowedTools)

        # Extract allowed tool names (mirrors streaming.py logic)
        allowed_tool_names = {
            tool["function"]["name"]
            for tool in result.allowed_tools.tools
            if tool.get("type") == "function" and "function" in tool
        }

        # Filter effective_tools (mirrors streaming.py logic)
        effective_tools = [tool for tool in chat_tools if tool.get("function", {}).get("name") in allowed_tool_names]

        # Should only have get_weather since nonexistent_tool isn't in chat_tools
        assert len(effective_tools) == 1
        assert effective_tools[0]["function"]["name"] == "get_weather"
