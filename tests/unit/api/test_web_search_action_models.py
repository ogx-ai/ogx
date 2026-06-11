# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import (
    OpenAIResponseOutputMessageWebSearchToolCall,
    WebSearchActionFind,
    WebSearchActionOpenPage,
    WebSearchActionSearch,
    WebSearchSource,
)


class TestWebSearchToolCallWithSearchAction:
    def test_search_action_with_sources(self):
        source = WebSearchSource(url="https://example.com/result1")
        action = WebSearchActionSearch(
            query="test query",
            queries=["test query", "related query"],
            sources=[source],
        )
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_001",
            status="completed",
            action=action,
        )

        assert tool_call.id == "ws_001"
        assert tool_call.status == "completed"
        assert tool_call.type == "web_search_call"
        assert tool_call.action is not None
        assert tool_call.action.type == "search"
        assert tool_call.action.query == "test query"
        assert tool_call.action.queries == ["test query", "related query"]
        assert len(tool_call.action.sources) == 1
        assert tool_call.action.sources[0].type == "url"
        assert tool_call.action.sources[0].url == "https://example.com/result1"


class TestWebSearchToolCallWithOpenPageAction:
    def test_open_page_action_with_url(self):
        action = WebSearchActionOpenPage(url="https://example.com/page")
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_002",
            status="completed",
            action=action,
        )

        assert tool_call.action is not None
        assert tool_call.action.type == "open_page"
        assert tool_call.action.url == "https://example.com/page"

    def test_open_page_action_without_url(self):
        action = WebSearchActionOpenPage()
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_003",
            status="in_progress",
            action=action,
        )

        assert tool_call.action is not None
        assert tool_call.action.type == "open_page"
        assert tool_call.action.url is None


class TestWebSearchToolCallWithFindAction:
    def test_find_action(self):
        action = WebSearchActionFind(
            url="https://example.com/page",
            pattern="search pattern",
        )
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_004",
            status="completed",
            action=action,
        )

        assert tool_call.action is not None
        assert tool_call.action.type == "find_in_page"
        assert tool_call.action.url == "https://example.com/page"
        assert tool_call.action.pattern == "search pattern"


class TestWebSearchToolCallWithoutAction:
    def test_backward_compat_no_action(self):
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_005",
            status="completed",
        )

        assert tool_call.id == "ws_005"
        assert tool_call.status == "completed"
        assert tool_call.type == "web_search_call"
        assert tool_call.action is None


class TestWebSearchToolCallSearchActionWithEmptySources:
    def test_search_action_empty_sources(self):
        action = WebSearchActionSearch(
            query="test query",
            sources=[],
        )
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_006",
            status="completed",
            action=action,
        )

        assert tool_call.action is not None
        assert tool_call.action.sources == []

    def test_search_action_none_sources(self):
        action = WebSearchActionSearch(query="test query")
        assert action.sources is None
        assert action.queries is None


class TestWebSearchToolCallSerialization:
    def test_serialization_with_search_action(self):
        source = WebSearchSource(url="https://example.com")
        action = WebSearchActionSearch(
            query="test",
            queries=["test"],
            sources=[source],
        )
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_007",
            status="completed",
            action=action,
        )
        data = tool_call.model_dump()

        assert data["id"] == "ws_007"
        assert data["status"] == "completed"
        assert data["type"] == "web_search_call"
        assert data["action"]["type"] == "search"
        assert data["action"]["query"] == "test"
        assert data["action"]["queries"] == ["test"]
        assert data["action"]["sources"] == [{"type": "url", "url": "https://example.com"}]

    def test_serialization_without_action(self):
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_008",
            status="completed",
        )
        data = tool_call.model_dump()

        assert data["id"] == "ws_008"
        assert data["action"] is None

    def test_serialization_with_open_page_action(self):
        action = WebSearchActionOpenPage(url="https://example.com/page")
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_009",
            status="completed",
            action=action,
        )
        data = tool_call.model_dump()

        assert data["action"]["type"] == "open_page"
        assert data["action"]["url"] == "https://example.com/page"

    def test_serialization_with_find_action(self):
        action = WebSearchActionFind(
            url="https://example.com/page",
            pattern="find this",
        )
        tool_call = OpenAIResponseOutputMessageWebSearchToolCall(
            id="ws_010",
            status="completed",
            action=action,
        )
        data = tool_call.model_dump()

        assert data["action"]["type"] == "find_in_page"
        assert data["action"]["url"] == "https://example.com/page"
        assert data["action"]["pattern"] == "find this"
