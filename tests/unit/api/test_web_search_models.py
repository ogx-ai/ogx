# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx_api.openai_responses import (
    OpenAIResponseInputToolWebSearch,
    WebSearchFilters,
    WebSearchUserLocation,
)


def test_web_search_tool_with_filters():
    tool = OpenAIResponseInputToolWebSearch(
        type="web_search",
        filters=WebSearchFilters(allowed_domains=["example.com", "docs.python.org"]),
    )
    assert tool.filters is not None
    assert tool.filters.allowed_domains == ["example.com", "docs.python.org"]


def test_web_search_tool_with_user_location():
    tool = OpenAIResponseInputToolWebSearch(
        type="web_search",
        user_location=WebSearchUserLocation(
            type="approximate",
            city="San Francisco",
            country="US",
            region="California",
            timezone="America/Los_Angeles",
        ),
    )
    assert tool.user_location is not None
    assert tool.user_location.type == "approximate"
    assert tool.user_location.city == "San Francisco"
    assert tool.user_location.country == "US"
    assert tool.user_location.region == "California"
    assert tool.user_location.timezone == "America/Los_Angeles"


def test_web_search_tool_with_all_fields():
    tool = OpenAIResponseInputToolWebSearch(
        type="web_search_preview",
        search_context_size="high",
        filters=WebSearchFilters(allowed_domains=["example.com"]),
        user_location=WebSearchUserLocation(
            type="approximate",
            city="London",
            country="GB",
            region="England",
            timezone="Europe/London",
        ),
    )
    assert tool.type == "web_search_preview"
    assert tool.search_context_size == "high"
    assert tool.filters is not None
    assert tool.filters.allowed_domains == ["example.com"]
    assert tool.user_location is not None
    assert tool.user_location.city == "London"


def test_web_search_tool_filters_none_by_default():
    tool = OpenAIResponseInputToolWebSearch(type="web_search")
    assert tool.filters is None
    assert tool.user_location is None
    assert tool.search_context_size is None


def test_web_search_tool_filters_with_empty_allowed_domains():
    filters = WebSearchFilters(allowed_domains=[])
    assert filters.allowed_domains == []


def test_web_search_tool_user_location_minimal():
    location = WebSearchUserLocation(type="approximate")
    assert location.type == "approximate"
    assert location.city is None
    assert location.country is None
    assert location.region is None
    assert location.timezone is None


def test_web_search_tool_invalid_search_context_size():
    with pytest.raises(ValidationError):
        OpenAIResponseInputToolWebSearch(
            type="web_search",
            search_context_size="extra_large",
        )
