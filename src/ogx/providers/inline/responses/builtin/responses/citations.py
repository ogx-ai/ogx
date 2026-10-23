# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Citations in Responses answers.

web_search results are listed to the model as `[N] title url`, numbered across the conversation. The model
cites `[N]`, and CitationRewriter replaces each cited number with ` ([domain](url))` and a url_citation spanning
that link. file_search keeps its `<|file-id|>` markers, which the same pass removes and turns into file_citations.
"""

import ipaddress
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from unicodedata import category, normalize
from urllib.parse import urlsplit

from ogx_api import (
    OpenAIResponseAnnotationCitation,
    OpenAIResponseAnnotationFileCitation,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageWebSearchToolCall,
    WebSearchActionSearch,
)

Annotation = OpenAIResponseAnnotationCitation | OpenAIResponseAnnotationFileCitation

CITATION_INSTRUCTION = (
    "When you use one of the numbered sources above, cite it with its number in square brackets at the end of "
    "the sentence, before the punctuation, like 'This is a fact [1].' Use only the numbers listed."
)

_MAX_URL_CHARS = 2048
_MAX_TITLE_CHARS = 200
# Percent-encoded in an inserted link destination, so it cannot end early, escape a character, split a table cell,
# close a code span, or leave an HTML attribute it was put in. Other characters stay for the renderer to encode.
_LINK_ESCAPED = re.compile(r"[()\[\]<>\"'\\|`]")
# Marker look-alikes, variation selectors, and letters and marks that render as nothing. Together with the C
# categories (tag characters and the ZWJ among them) this leaves a title no hidden text.
_TITLE_DROPPED = frozenset(
    [
        *"[]<>【】\u034f\u115f\u1160\u3164\uffa0\u2800",
        *map(chr, range(0xFE00, 0xFE10)),
        *map(chr, range(0xE0100, 0xE01F0)),
    ]
)

_NUMBER = r"[1-9]\d{0,3}"
_NUMBERS = rf"{_NUMBER}(?:, ?{_NUMBER}){{0,5}}"
_FILE_ID = r"file-[A-Za-z0-9_-]{1,64}"
# A file marker <|id|>, [id] or (id), with the space before it and any parentheses around it. It counts in code too.
_FILE = (
    rf" ?(?P<paren>\()?(?:<\|(?P<pipe>{_FILE_ID})\|>|\[(?P<bracket>{_FILE_ID})\]|\((?P<round>{_FILE_ID})\))(?(paren)\))"
)
# Outside code: a backtick run that opens code, a web marker [N] or [N, M], or a file marker. A web marker followed
# by "(" (a link), "=", or a letter or digit is not one.
_TOKEN = re.compile(rf"(?P<ticks>(?<!`)`+)|\[(?P<numbers>{_NUMBERS})\](?![(A-Za-z0-9]| ?=)|{_FILE}")
_FILE_PREFIX = r"(?:f(?:i(?:l(?:e(?:-[A-Za-z0-9_-]{0,64})?)?)?)?)?"
# A tail that more text could still turn into a token, or that the next character decides (after a [N]). It is
# held back until that is settled.
_HOLD = re.compile(
    r"(?<!`)`{1,2}\Z"
    rf"| ?(?:\[{_NUMBERS}\] ?)?"
    rf"(?:\(?(?:<(?:\|{_FILE_PREFIX}(?:\|>?)?)?|\[{_FILE_PREFIX}\]?|\({_FILE_PREFIX}\)?)"
    rf"|\[(?:{_NUMBER}(?:, ?(?:{_NUMBER})?){{0,5}}(?:\] ?)?)?)?\Z"
)
# Inside code opened by a run of 1, 2 or 3+ backticks: what ends it, or a file marker.
_CODE = {
    1: re.compile(rf"(?:(?<!`)`(?!`)|\n)|{_FILE}"),
    2: re.compile(rf"(?:(?<!`)``(?!`)|\n)|{_FILE}"),
    3: re.compile(rf"(?<!`)`{{3,}}|{_FILE}"),
}


def valid_url(url: object) -> str | None:
    """url, unchanged, if it is a printable http(s) URL without spaces or angle brackets, whose host is an IPv6
    address or made of letters, marks, digits, "-" and "_", else None."""
    if not isinstance(url, str) or len(url) > _MAX_URL_CHARS or not url.isprintable() or any(c in url for c in " <>"):
        return None
    try:
        parts = urlsplit(url)
        host, _port = parts.hostname or "", parts.port
        if "[" in parts.netloc:
            ipaddress.IPv6Address(host)  # not an IPvFuture literal such as [v1.x]
        else:
            host.encode("idna")  # only to reject empty and over-long labels
    except ValueError:
        return None
    # "%" in the host is an IPv6 zone id, which ipaddress accepts, or an encoded host.
    if parts.scheme not in ("http", "https") or not host or len(host) > 253 or "@" in parts.netloc or "%" in host:
        return None
    return url if "[" in parts.netloc or all(c in ".-_" or category(c)[0] in "LMN" for c in host) else None


def link_label(url: str) -> str:
    """The domain an inserted link shows, in punycode, so a look-alike host cannot pass for another. NFKC maps it
    to the host a browser resolves, as for full-width letters."""
    host = normalize("NFKC", urlsplit(url).hostname or "").lower().removeprefix("www.")
    return ".".join(part if part.isascii() else "xn--" + part.encode("punycode").decode() for part in host.split("."))


def sanitize_title(title: object, url: str) -> str:
    """title as one line of visible text with no marker look-alikes, else the URL's domain."""
    text = " ".join(title.split()) if isinstance(title, str) else ""
    kept = "".join(c for c in text if c == " " or (c not in _TITLE_DROPPED and category(c)[0] != "C"))
    return " ".join(kept.split())[:_MAX_TITLE_CHARS].strip() or link_label(url)


@dataclass(frozen=True, slots=True)
class WebSource:
    """A web_search result: its URL as the tool returned it, and a sanitised title."""

    url: str
    title: str


class CitationRegistry:
    """The web sources of one conversation, under the numbers the model sees.

    file_search labels its chunks [1], [2], ... in each search until the conversation has a web number. From then
    on its labels start above every number shown so far, and new web numbers always do, so no number names two
    things.
    """

    def __init__(self) -> None:
        self.numbers: dict[str, int] = {}
        self.sources: dict[int, WebSource] = {}
        self.shown = 0

    def label_chunks(self, count: int) -> int:
        """The first of the labels for one search's count chunks."""
        first = self.shown + 1 if self.numbers else 1
        self.shown = max(self.shown, first + count - 1)
        return first

    def add(self, url: object, title: object) -> int | None:
        if (checked := valid_url(url)) is None:
            return None
        number = self.numbers.get(checked)
        if number is None:
            number = self.numbers[checked] = self.shown = self.shown + 1
        elif self.sources[number].title != link_label(checked):
            return number  # the first real title stays; a domain fallback gives way to one
        self.sources[number] = WebSource(checked, sanitize_title(title, checked))
        return number

    def replay(self, items: Iterable[object], stored: bool) -> None:
        """Number the sources of earlier web_search_call items, and the chunks of earlier file_search_call items
        (one result per chunk). Only stored history gives titles, from its url_citations; a client can send any
        title."""
        items = list(items)
        titles = _url_citation_titles(items) if stored else {}
        for item in items:
            if isinstance(item, OpenAIResponseOutputMessageFileSearchToolCall):
                self.label_chunks(len(item.results or []))
            elif isinstance(item, OpenAIResponseOutputMessageWebSearchToolCall) and isinstance(
                item.action, WebSearchActionSearch
            ):
                for source in item.action.sources or []:
                    self.add(source.url, titles.get(source.url))

    def source_list(self, numbers: Iterable[int]) -> str:
        return "Sources:\n" + "\n".join(f"[{n}] {self.sources[n].title} {self.sources[n].url}" for n in numbers)


def _url_citation_titles(items: list[object]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for item in items:
        if isinstance(item, OpenAIResponseMessage) and isinstance(item.content, list):
            for part in item.content:
                for annotation in getattr(part, "annotations", None) or []:
                    if isinstance(annotation, OpenAIResponseAnnotationCitation):
                        titles.setdefault(annotation.url, annotation.title)
    return titles


class CitationRewriter:
    """Rewrites one streamed answer. Web markers stay text inside backtick code spans and fences; file markers do not.

    files maps the file ids file_search returned in this response to their filenames, best first. sources is
    None when web citations are off.
    """

    def __init__(self, files: Mapping[str, str], sources: CitationRegistry | None = None) -> None:
        self._files = files
        self._sources = sources
        self.annotations: list[Annotation] = []
        self._out: list[str] = []
        self._out_len = 0
        # Raw text not yet rewritten starts at _start in _buf, after one character of context. _offset is where
        # _buf starts in the raw answer.
        self._buf = ""
        self._start = 0
        self._offset = 0
        # The length of the backtick run that opened the code we are in (3 for any fence), else 0.
        self._code = 0
        # Where the last [N] left as text ends, so "[9][1]" and "x[1][2]" stay text whole.
        self._text_end = -1

    @property
    def text(self) -> str:
        return "".join(self._out)

    def feed(self, delta: str) -> tuple[str, int]:
        """Return the text that became final, and the index in annotations of the first one it completed."""
        self._buf += delta
        hold = _HOLD.search(self._buf, self._start)
        return self._rewrite(hold.start() if hold else len(self._buf))

    def finish(self) -> tuple[str, int]:
        """Like feed() for the held tail. If nothing was cited, the best file file_search returned is."""
        step = self._rewrite(len(self._buf))
        if not self.annotations and self._files:
            file_id, filename = next(iter(self._files.items()))
            self.annotations.append(
                OpenAIResponseAnnotationFileCitation(file_id=file_id, filename=filename, index=self._out_len)
            )
        return step

    def _rewrite(self, end: int) -> tuple[str, int]:
        first_out, first_annotation = len(self._out), len(self.annotations)
        buf, i = self._buf, self._start
        while i < end:
            match = (_CODE[self._code] if self._code else _TOKEN).search(buf, i, end)
            if match is None:
                self._emit(buf[i:end])
                break
            self._emit(buf[i : match.start()])
            if match["pipe"] or match["bracket"] or match["round"]:
                self._file(match)
            elif self._code:
                self._code = 0
                self._emit(match[0])
            elif match["numbers"]:
                self._web(match)
            else:
                self._code = min(len(match["ticks"]), 3)
                self._emit(match[0])
            i = match.end()
        keep = max(end - 1, 0)
        self._buf, self._start, self._offset = buf[keep:], end - keep, self._offset + keep
        return "".join(self._out[first_out:]), first_annotation

    def _web(self, match: re.Match[str]) -> None:
        before = self._buf[match.start() - 1 : match.start()]
        glued = (before.isascii() and before.isalnum()) or (
            before == "]" and self._offset + match.start() == self._text_end
        )
        numbers = [int(n) for n in match["numbers"].split(",")]
        sources = [s for n in numbers if self._sources and (s := self._sources.sources.get(n))]
        if glued or len(sources) < len(numbers):
            self._text_end = self._offset + match.end()
            self._emit(match[0])
            return
        for k, source in enumerate(sources):
            if k or (self._out and not self._out[-1][-1].isspace()):
                self._emit(" ")
            # An escaped "&" cannot start a character reference.
            target = _LINK_ESCAPED.sub(lambda c: f"%{ord(c[0]):02X}", source.url).replace("&", "\\&")
            link = f"([{link_label(source.url)}]({target}))"
            self.annotations.append(
                OpenAIResponseAnnotationCitation(
                    start_index=self._out_len,
                    end_index=self._out_len + len(link),
                    url=source.url,
                    title=source.title,
                )
            )
            self._emit(link)

    def _file(self, match: re.Match[str]) -> None:
        file_id = match["pipe"] or match["bracket"] or match["round"]
        filename = self._files.get(file_id)
        if filename is None:
            self._emit(match[0])
            return
        self.annotations.append(
            OpenAIResponseAnnotationFileCitation(file_id=file_id, filename=filename, index=self._out_len)
        )

    def _emit(self, text: str) -> None:
        if text:
            self._out.append(text)
            self._out_len += len(text)
