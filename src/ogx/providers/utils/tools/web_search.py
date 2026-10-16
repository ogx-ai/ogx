# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared shape for web_search tool_runtime providers' result metadata.

Each web_search provider (bing, brave, exa, nimble, serply, tavily) returns
ToolInvocationResult.metadata["sources"]: a list of source dicts, read independently by the
Responses layer to build both web_search_call.action.sources and url_citation annotations
(see tool_executor.py). Since metadata is an untyped dict[str, Any], a shared TypedDict and
parser keep providers and both readers from drifting apart on the shape or on how a malformed
entry gets handled.
"""

from typing import Any, TypedDict

from pydantic import HttpUrl, TypeAdapter, ValidationError

from ogx.log import get_logger

logger = get_logger(name=__name__, category="tool_runtime")

# HttpUrl checks scheme (http/https only, so javascript:/data: can never pass) and host,
# percent-encodes spaces, control characters (including \u2028/\x85, which a naive character
# denylist missed) and <, >, caps length at 2083, and allows everything else a real url can
# legitimately contain (e.g. a query string with a literal "|").
_HTTP_URL = TypeAdapter(HttpUrl)


class WebSearchSourceDict(TypedDict):
    """One web_search result source, as providers should build it for
    ToolInvocationResult.metadata["sources"]."""

    url: str
    title: str


def _collapse_whitespace(title: str, url: str) -> str:
    """Collapse whitespace (including newlines) to single spaces, so a title can't fake a
    second numbered source line in the citation instruction text (see
    _build_web_search_citation_block). Falls back to the url if that leaves nothing.

    Citation-marker-looking substrings (`<|id|>`, `[id]`, `(id)`) are a separate concern,
    handled where the source line is built (tool_executor.py), not here: that needs
    CITATION_MARKER_REGEX, which lives with the rest of the marker machinery in utils.py, not
    in this general-purpose parser.
    """
    collapsed = " ".join(title.split())
    return collapsed or url


def parse_web_search_sources(raw_sources: Any) -> list[WebSearchSourceDict]:
    """Validate and parse a raw metadata["sources"] value into well-formed WebSearchSourceDicts.

    Providers are external services; a malformed entry (metadata["sources"] itself not a
    list, an element that isn't a dict, or an invalid url) is skipped and logged, not raised,
    since raising here would otherwise fail a web_search call that already succeeded (see
    tool_executor.py, which builds citations from this after the tool call completed).

    Also enforces http(s)-only, well-formed urls via HttpUrl (see _HTTP_URL) and collapses
    title whitespace (see _collapse_whitespace), since both are rendered into text a client or
    model reads as if they were plain facts, not attacker-influenced content.
    """
    if not isinstance(raw_sources, list):
        if raw_sources:
            logger.warning("Ignoring non-list web_search sources metadata", sources=repr(raw_sources))
        return []

    parsed: list[WebSearchSourceDict] = []
    for source in raw_sources:
        if not isinstance(source, dict):
            logger.warning("Skipping malformed web_search source: not a dict", source=repr(source))
            continue
        try:
            url = str(_HTTP_URL.validate_python(source.get("url")))
        except ValidationError:
            logger.warning("Skipping web_search source with an invalid url", source=repr(source))
            continue
        title = source.get("title")
        if not isinstance(title, str) or not title:
            title = url
        parsed.append(WebSearchSourceDict(url=url, title=_collapse_whitespace(title, url)))
    return parsed
