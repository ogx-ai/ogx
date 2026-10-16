# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.inline.responses.builtin.responses.types import CitationUrl
from ogx.providers.inline.responses.builtin.responses.utils import (
    _extract_citations_from_text,
    convert_chat_choice_to_response_message,
    extract_citations_from_text,
)
from ogx_api.inference import OpenAIChatCompletionResponseMessage, OpenAIChoice
from ogx_api.openai_responses import (
    OpenAIResponseAnnotationCitation,
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseOutputMessageContentOutputText,
)


class TestExtractUrlCitationsFromText:
    """Regression tests for #6763: web_search never produced url_citation annotations."""

    def test_extracts_url_citation_with_span_limited_to_its_own_sentence(self):
        """The second citation's span starts at its own sentence ("They were introduced in
        2017"), not all the way back at the checkpoint left by the first citation -- the two
        sentences are both attributed to web_search, but each citation covers only the
        sentence it's actually attached to.
        """
        text = "Transformers use self-attention <|url-1|>. They were introduced in 2017 <|url-2|>."
        citation_urls = {
            "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need"),
            "url-2": CitationUrl(url="https://en.wikipedia.org/wiki/Transformer", title="Transformer"),
        }

        annotations, cleaned_text = _extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == "Transformers use self-attention. They were introduced in 2017."
        assert annotations == [
            OpenAIResponseAnnotationCitation(
                url="https://arxiv.org/abs/1706.03762",
                title="Attention Is All You Need",
                start_index=0,
                end_index=31,
            ),
            OpenAIResponseAnnotationCitation(
                url="https://en.wikipedia.org/wiki/Transformer",
                title="Transformer",
                start_index=33,
                end_index=61,
            ),
        ]
        assert cleaned_text[annotations[0].end_index] == "."
        assert cleaned_text[annotations[1].end_index] == "."
        assert cleaned_text[annotations[1].start_index :] == "They were introduced in 2017."

    def test_preserves_unknown_url_marker(self):
        text = "Some fact <|url-99|>."
        annotations, cleaned_text = _extract_citations_from_text(text, {}, {"url-1": CitationUrl(url="u", title="t")})

        assert annotations == []
        assert cleaned_text == "Some fact <|url-99|>."

    def test_adjacent_url_markers_share_the_same_span_instead_of_one_being_empty(self):
        """Regression test: a model citing one claim with several sources
        ("fact <|url-1|><|url-2|>.") glues the markers together with no text between
        them. Giving the second marker a span starting where the first one ended would
        make it a zero-width (x, x) span pointing at nothing; both should instead cover
        the same real preceding text.
        """
        text = "Transformers use self-attention <|url-1|><|url-2|>."
        citation_urls = {
            "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need"),
            "url-2": CitationUrl(url="https://en.wikipedia.org/wiki/Transformer", title="Transformer"),
        }

        annotations, cleaned_text = _extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == "Transformers use self-attention."
        assert annotations == [
            OpenAIResponseAnnotationCitation(
                url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need", start_index=0, end_index=31
            ),
            OpenAIResponseAnnotationCitation(
                url="https://en.wikipedia.org/wiki/Transformer", title="Transformer", start_index=0, end_index=31
            ),
        ]

    def test_three_adjacent_url_markers_all_share_the_same_span(self):
        text = "Fact <|url-1|><|url-2|><|url-3|>."
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
            "url-3": CitationUrl(url="https://c.com", title="C"),
        }

        annotations, cleaned_text = _extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == "Fact."
        assert [(a.start_index, a.end_index) for a in annotations] == [(0, 4), (0, 4), (0, 4)]

    def test_url_marker_after_an_adjacent_cluster_gets_its_own_fresh_span(self):
        """url-3's span covers only "Fact B", its own sentence, not the space separating it
        from the previous cluster's sentence.
        """
        text = "Fact A <|url-1|><|url-2|> Fact B <|url-3|>."
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
            "url-3": CitationUrl(url="https://c.com", title="C"),
        }

        annotations, cleaned_text = _extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == "Fact A Fact B."
        assert (annotations[0].start_index, annotations[0].end_index) == (0, 6)
        assert (annotations[1].start_index, annotations[1].end_index) == (0, 6)
        assert (annotations[2].start_index, annotations[2].end_index) == (7, 13)
        assert cleaned_text[7:13] == "Fact B"

    def test_file_and_url_markers_coexist_with_correct_indices(self):
        """A single turn can call both file_search and web_search; markers of both
        types must be strippable in one pass without corrupting each other's indices.

        The url citation's span starts at its own sentence ("From the web"), not at the very
        beginning of the text and not merely right after the preceding file citation: "From a
        doc" was already attributed to the file citation, so the url citation must not also
        claim it or the sentence-boundary whitespace between the two.
        """
        text = "From a doc <|file-abc123|>. From the web <|url-1|>."
        citation_files = {"file-abc123": "doc1.pdf"}
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = _extract_citations_from_text(text, citation_files, citation_urls)

        assert cleaned_text == "From a doc. From the web."
        assert annotations == [
            OpenAIResponseAnnotationFileCitation(file_id="file-abc123", filename="doc1.pdf", index=10),
            OpenAIResponseAnnotationCitation(url="https://example.com", title="Example", start_index=12, end_index=24),
        ]
        assert cleaned_text[12:24] == "From the web"

    def test_url_marker_glued_to_file_marker_shares_the_same_cluster_start(self):
        """`<|url-1|><|file-a|><|url-2|>`, with no text separating any of the three markers,
        is one claim backed by two sources; url-2 must share url-1's start instead of getting
        a zero-width span once the file citation sits between them.
        """
        text = "Fact <|url-1|><|file-a|><|url-2|>."
        citation_files = {"file-a": "doc.pdf"}
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
        }

        annotations, cleaned_text = _extract_citations_from_text(text, citation_files, citation_urls)

        assert cleaned_text == "Fact."
        url_annotations = [a for a in annotations if isinstance(a, OpenAIResponseAnnotationCitation)]
        assert [(a.start_index, a.end_index) for a in url_annotations] == [(0, 4), (0, 4)]

    def test_unknown_markers_between_a_known_marker_and_its_sentence_start_dont_break_the_span(self):
        """Regression test for the seg_start/prefix_end deferred-recompute optimization: an
        unknown/stale marker inside the sentence a later known marker is attached to must not
        get folded into the join and corrupt the sentence-start scan -- the sentence-start
        rescan is deferred until a known marker is actually reached, but it must still cover
        the full span of text accumulated since the last checkpoint, unknown markers included.
        """
        text = "Fact one. Unverified claim <|url-fake|> about fact two <|url-1|>."
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = _extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == "Fact one. Unverified claim <|url-fake|> about fact two."
        assert len(annotations) == 1
        assert (annotations[0].start_index, annotations[0].end_index) == (10, 54)
        assert cleaned_text[10:54] == "Unverified claim <|url-fake|> about fact two"

    def test_many_unknown_markers_do_not_cause_quadratic_slowdown(self):
        """Regression test for a DoS finding: the sentence-start rescan used to re-join and
        re-scan the *entire* text accumulated so far on every single marker, including
        unknown ones (which never advance the checkpoint and are fully attacker-controlled,
        since they're never validated against a real id) -- making a long run of them
        quadratic in the number of markers. 20,000 repeated fake markers previously took on
        the order of 10+ seconds; this must stay well under a second.
        """
        import time

        text = "<|url-fake|>" * 20000

        start = time.perf_counter()
        annotations, _ = _extract_citations_from_text(text, {}, {})
        elapsed = time.perf_counter() - start

        assert annotations == []
        assert elapsed < 2.0

    def test_falls_back_to_whole_text_span_when_model_does_not_cite(self):
        text = "Transformers use self-attention."
        citation_urls = {
            "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need")
        }

        annotations, cleaned_text = extract_citations_from_text(text, {}, citation_urls)

        assert cleaned_text == text
        assert len(annotations) == 1
        assert annotations[0].url == "https://arxiv.org/abs/1706.03762"
        assert annotations[0].start_index == 0
        assert annotations[0].end_index == len(text)

    def test_fallback_cites_only_first_source(self):
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
        }

        annotations, _ = extract_citations_from_text("Some text.", {}, citation_urls)

        assert len(annotations) == 1
        assert annotations[0].url == "https://a.com"

    def test_no_fallback_when_no_urls_were_retrieved(self):
        annotations, cleaned_text = extract_citations_from_text("No sources used here.", {}, {})

        assert annotations == []
        assert cleaned_text == "No sources used here."

    def test_real_marker_citation_takes_precedence_over_url_fallback(self):
        text = "Cited fact <|url-1|>. Uncited fact."
        citation_urls = {
            "url-1": CitationUrl(url="https://a.com", title="A"),
            "url-2": CitationUrl(url="https://b.com", title="B"),
        }

        annotations, cleaned_text = extract_citations_from_text(text, {}, citation_urls)

        assert len(annotations) == 1
        assert annotations[0].url == "https://a.com"
        assert cleaned_text == "Cited fact. Uncited fact."

    def test_both_fallbacks_apply_when_nothing_was_cited_at_all(self):
        """When the model cites nothing inline at all, both a file_citation and a
        url_citation fallback should be produced -- one per retrieval type."""
        text = "An answer that draws on both sources."
        citation_files = {"file-abc123": "doc1.pdf"}
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = extract_citations_from_text(text, citation_files, citation_urls)

        assert len(annotations) == 2
        assert isinstance(annotations[0], OpenAIResponseAnnotationFileCitation)
        assert isinstance(annotations[1], OpenAIResponseAnnotationCitation)
        assert annotations[0].index == len(cleaned_text)
        assert annotations[1].end_index == len(cleaned_text)

    def test_no_url_fallback_when_the_model_already_cited_a_file_inline(self):
        """Regression test: a model that cites a real file_citation inline made a
        deliberate attribution choice. Guessing an extra whole-answer url_citation for
        web_search results it didn't cite would misrepresent that choice, even though
        file_search's own fallback check (now looking for *any* citation, not just a
        file one) correctly stays silent too.
        """
        text = "Cited from the doc <|file-abc123|>."
        citation_files = {"file-abc123": "doc1.pdf"}
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = extract_citations_from_text(text, citation_files, citation_urls)

        assert cleaned_text == "Cited from the doc."
        assert len(annotations) == 1
        assert isinstance(annotations[0], OpenAIResponseAnnotationFileCitation)
        assert annotations[0].file_id == "file-abc123"

    def test_no_file_fallback_when_the_model_already_cited_a_url_inline(self):
        """Mirror of the above: a real inline url_citation should suppress the file
        fallback too, when file_search also ran but wasn't cited."""
        text = "Cited from the web <|url-1|>."
        citation_files = {"file-abc123": "doc1.pdf"}
        citation_urls = {"url-1": CitationUrl(url="https://example.com", title="Example")}

        annotations, cleaned_text = extract_citations_from_text(text, citation_files, citation_urls)

        assert cleaned_text == "Cited from the web."
        assert len(annotations) == 1
        assert isinstance(annotations[0], OpenAIResponseAnnotationCitation)
        assert annotations[0].url == "https://example.com"


class TestConvertChatChoiceToResponseMessageWithUrlCitations:
    async def test_strips_url_marker_and_attaches_url_citation(self):
        choice = OpenAIChoice(
            message=OpenAIChatCompletionResponseMessage(content="Transformers use self-attention <|url-1|>."),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(
            choice,
            citation_urls={
                "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need")
            },
        )

        assert isinstance(result.content[0], OpenAIResponseOutputMessageContentOutputText)
        assert result.content[0].text == "Transformers use self-attention."
        assert len(result.content[0].annotations) == 1
        annotation = result.content[0].annotations[0]
        assert annotation.url == "https://arxiv.org/abs/1706.03762"
        assert annotation.title == "Attention Is All You Need"

    async def test_populates_fallback_url_citation_when_model_does_not_cite_inline(self):
        choice = OpenAIChoice(
            message=OpenAIChatCompletionResponseMessage(content="Transformers use self-attention."),
            finish_reason="stop",
            index=0,
        )

        result = await convert_chat_choice_to_response_message(
            choice,
            citation_urls={
                "url-1": CitationUrl(url="https://arxiv.org/abs/1706.03762", title="Attention Is All You Need")
            },
        )

        assert len(result.content[0].annotations) == 1
        assert result.content[0].annotations[0].url == "https://arxiv.org/abs/1706.03762"
