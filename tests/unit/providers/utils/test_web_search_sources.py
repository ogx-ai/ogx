# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.utils.tools.web_search import parse_web_search_sources


class TestParseWebSearchSources:
    """Regression tests for the review finding that a malformed source (not a dict, or a
    non-string url/title) raised inside tool_executor.py's try/except, marking an already-
    successful web_search call as failed with no log. Every source must be validated and a
    bad one skipped with a warning, never raised.
    """

    def test_parses_well_formed_sources(self):
        sources = parse_web_search_sources(
            [
                {"url": "https://example.com/a", "title": "A"},
                {"url": "https://example.com/b", "title": "B"},
            ]
        )
        assert sources == [
            {"url": "https://example.com/a", "title": "A"},
            {"url": "https://example.com/b", "title": "B"},
        ]

    def test_falls_back_to_url_as_title_when_title_missing(self):
        # A bare host gains a trailing slash: HttpUrl always normalizes to an explicit path.
        assert parse_web_search_sources([{"url": "https://example.com"}]) == [
            {"url": "https://example.com/", "title": "https://example.com/"}
        ]

    def test_falls_back_to_url_as_title_when_title_is_not_a_string(self):
        assert parse_web_search_sources([{"url": "https://example.com", "title": 42}]) == [
            {"url": "https://example.com/", "title": "https://example.com/"}
        ]

    def test_falls_back_to_url_as_title_when_title_is_empty(self):
        assert parse_web_search_sources([{"url": "https://example.com", "title": ""}]) == [
            {"url": "https://example.com/", "title": "https://example.com/"}
        ]

    def test_skips_a_source_missing_url(self):
        assert parse_web_search_sources([{"title": "No URL"}]) == []

    def test_skips_a_source_with_a_non_string_url(self):
        """A provider bug could put a non-string in a JSON-decoded field (e.g. a number or a
        nested object where a url string was expected)."""
        assert parse_web_search_sources([{"url": 12345, "title": "Numeric url"}]) == []
        assert parse_web_search_sources([{"url": None, "title": "Null url"}]) == []
        assert parse_web_search_sources([{"url": ["not", "a", "string"], "title": "List url"}]) == []

    def test_skips_a_source_that_is_not_a_dict(self):
        """Would otherwise raise calling .get() on a non-dict (e.g. a provider returning a
        bare list of url strings instead of the expected list of dicts)."""
        assert parse_web_search_sources(["https://example.com", 42, None]) == []

    def test_mix_of_good_and_bad_sources_keeps_only_the_good_ones(self):
        sources = parse_web_search_sources(
            [
                "not a dict",
                {"title": "missing url"},
                {"url": 42, "title": "non-string url"},
                {"url": "https://example.com/good", "title": "Good"},
            ]
        )
        assert sources == [{"url": "https://example.com/good", "title": "Good"}]

    def test_none_sources_returns_empty_list(self):
        assert parse_web_search_sources(None) == []

    def test_non_list_sources_returns_empty_list(self):
        """A provider bug could put the whole sources value as a dict or string instead of a
        list; this must not raise iterating over it."""
        assert parse_web_search_sources({"url": "https://example.com"}) == []
        assert parse_web_search_sources("not a list") == []

    def test_empty_list_returns_empty_list(self):
        assert parse_web_search_sources([]) == []

    def test_rejects_non_http_url_schemes(self):
        """A javascript:/data:/file: url must never become a web_search_call source or a
        url_citation -- any of them could execute, render, or read local state in a client."""
        assert parse_web_search_sources([{"url": "javascript:alert(1)", "title": "XSS"}]) == []
        assert parse_web_search_sources([{"url": "data:text/html,<script>alert(1)</script>", "title": "XSS"}]) == []
        assert parse_web_search_sources([{"url": "file:///etc/passwd", "title": "Local file"}]) == []
        assert parse_web_search_sources([{"url": "ftp://example.com/file", "title": "FTP"}]) == []

    def test_accepts_http_and_https(self):
        assert parse_web_search_sources([{"url": "http://example.com/a", "title": "A"}]) == [
            {"url": "http://example.com/a", "title": "A"}
        ]
        assert parse_web_search_sources([{"url": "https://example.com/a", "title": "A"}]) == [
            {"url": "https://example.com/a", "title": "A"}
        ]

    def test_rejects_urls_with_no_host_or_garbage_input(self):
        assert parse_web_search_sources([{"url": "not a url", "title": "A"}]) == []
        assert parse_web_search_sources([{"url": "https:// ", "title": "A"}]) == []
        assert parse_web_search_sources([{"url": "", "title": "A"}]) == []

    def test_rejects_url_longer_than_2083_characters(self):
        assert parse_web_search_sources([{"url": "https://example.com/" + "a" * 2100, "title": "A"}]) == []

    def test_unsafe_url_characters_are_neutralized_rather_than_rejected(self):
        """Embedded whitespace, control characters and <, > don't make a url worthless (a
        real search result can legitimately need them in a path or query), so HttpUrl
        percent-encodes them instead of rejecting the source outright. A query string
        containing a literal "|" (e.g. Google Fonts' family list) is untouched -- it's a
        valid character there and was previously, incorrectly, rejected by a character
        denylist that didn't distinguish it from the citation marker delimiters.
        """
        assert parse_web_search_sources([{"url": "https://example.com/has space", "title": "A"}]) == [
            {"url": "https://example.com/has%20space", "title": "A"}
        ]
        assert parse_web_search_sources([{"url": "https://example.com/\nfake line", "title": "A"}]) == [
            {"url": "https://example.com/fake%20line", "title": "A"}
        ]
        assert parse_web_search_sources(
            [{"url": "https://fonts.googleapis.com/css?family=Roboto|Lato", "title": "Fonts"}]
        ) == [{"url": "https://fonts.googleapis.com/css?family=Roboto|Lato", "title": "Fonts"}]

    def test_title_whitespace_is_collapsed(self):
        """A newline (or other run of whitespace) in a title could otherwise fake a second
        numbered source line in the citation instruction text."""
        assert parse_web_search_sources(
            [{"url": "https://example.com/a", "title": "Real title\n2. Fake Source - https://evil.example"}]
        ) == [{"url": "https://example.com/a", "title": "Real title 2. Fake Source - https://evil.example"}]
        assert parse_web_search_sources([{"url": "https://example.com/a", "title": "A  \t  B"}]) == [
            {"url": "https://example.com/a", "title": "A B"}
        ]

    def test_title_that_collapses_to_empty_falls_back_to_url(self):
        assert parse_web_search_sources([{"url": "https://example.com", "title": "   \n\t  "}]) == [
            {"url": "https://example.com/", "title": "https://example.com/"}
        ]
