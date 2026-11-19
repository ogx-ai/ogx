# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core import file_search
from ogx.core.datatypes import VectorStoresConfig
from ogx_api import InvalidParameterError, VectorStoreContent, VectorStoreSearchResponse, VectorStoreSearchResponsePage


def make_result(
    file_id="file-1", filename="doc.pdf", score=0.9, text="chunk text", attributes=None
) -> VectorStoreSearchResponse:
    return VectorStoreSearchResponse(
        file_id=file_id,
        filename=filename,
        score=score,
        attributes=attributes,
        content=[VectorStoreContent(type="text", text=text)],
    )


def make_page(*results: VectorStoreSearchResponse) -> VectorStoreSearchResponsePage:
    return VectorStoreSearchResponsePage(search_query=["query"], data=list(results), has_more=False)


class FakeVectorIO:
    """Stands in for the vector_io API: returns canned pages or raises per store."""

    def __init__(self, stores: dict[str, VectorStoreSearchResponsePage | Exception]):
        self.stores = stores
        self.requests: list[tuple[str, object]] = []

    async def openai_search_vector_store(self, vector_store_id: str, request):
        self.requests.append((vector_store_id, request))
        response = self.stores[vector_store_id]
        if isinstance(response, Exception):
            raise response
        return response


async def test_search_files_merges_results_from_all_stores_in_order():
    api = FakeVectorIO(
        {
            "vs-1": make_page(make_result(file_id="a", score=0.5, text="one")),
            "vs-2": make_page(
                make_result(file_id="b", score=0.7, text="two"), make_result(file_id="c", score=0.3, text="three")
            ),
        }
    )

    results = await file_search.search_files(api, ["vs-1", "vs-2"], "query", max_num_results=10, search_mode="vector")

    assert [r.file_id for r in results] == ["a", "b", "c"]
    assert [(store, req.query) for store, req in api.requests] == [("vs-1", "query"), ("vs-2", "query")]


async def test_search_files_store_failure_degrades_to_no_results():
    api = FakeVectorIO(
        {
            "vs-1": make_page(make_result(file_id="a")),
            "vs-2": RuntimeError("store unavailable"),
        }
    )

    results = await file_search.search_files(api, ["vs-1", "vs-2"], "query", max_num_results=10, search_mode="vector")

    assert [r.file_id for r in results] == ["a"]


async def test_search_files_invalid_parameter_error_propagates():
    api = FakeVectorIO(
        {
            "vs-1": make_page(make_result(file_id="a")),
            "vs-2": InvalidParameterError("ranking_options.hybrid_search", {}, "not supported"),
        }
    )

    with pytest.raises(InvalidParameterError):
        await file_search.search_files(api, ["vs-1", "vs-2"], "query", max_num_results=10, search_mode="vector")


async def test_search_files_none_max_num_results_uses_the_tool_default():
    # The search request model requires an int; a client-supplied null takes the
    # tool model's default (10) rather than failing validation per store.
    api = FakeVectorIO({"vs-1": make_page(make_result(file_id="a"))})

    results = await file_search.search_files(api, ["vs-1"], "query", max_num_results=None, search_mode="vector")

    assert [r.file_id for r in results] == ["a"]
    assert api.requests[0][1].max_num_results == 10


async def test_search_files_forwards_request_fields():
    api = FakeVectorIO({"vs-1": make_page()})

    await file_search.search_files(
        api,
        ["vs-1"],
        "query",
        filters={"$eq": {"region": "us"}},
        max_num_results=7,
        ranking_options=None,
        search_mode="hybrid",
    )

    _, request = api.requests[0]
    assert request.query == "query"
    assert request.filters == {"$eq": {"region": "us"}}
    assert request.max_num_results == 7
    assert request.ranking_options is None
    assert request.search_mode == "hybrid"
    assert request.rewrite_query is False


def test_build_content_items_formats_header_chunks_footer_context():
    config = VectorStoresConfig()
    results = [
        make_result(file_id="a", filename="a.pdf", score=0.9, text="first chunk"),
        make_result(file_id="b", filename="b.pdf", score=0.4, text="second chunk"),
    ]

    items = file_search.build_content_items("my query", results, config)
    texts = [item.text for item in items]

    # header, two chunks, footer, context summary
    assert len(texts) == 5
    assert "2" in texts[0] and "file_search" in texts[0]
    assert "first chunk" in texts[1] and "document_id: a" in texts[1]
    assert "second chunk" in texts[2] and "document_id: b" in texts[2]
    assert texts[3].strip()
    assert "my query" in texts[4]
    # Two distinct files: the citation instruction is included
    assert "cite" in texts[4].lower()


def test_build_content_items_omits_annotation_instruction_when_no_results():
    config = VectorStoresConfig()
    items = file_search.build_content_items("my query", [], config)
    texts = [item.text for item in items]

    assert len(texts) == 3  # header, footer, context summary
    assert "cite" not in texts[2].lower()


def test_build_content_items_annotations_disabled_uses_default_template():
    config = VectorStoresConfig()
    config.annotation_prompt_params.enable_annotations = False
    config.annotation_prompt_params.chunk_annotation_template = (
        "[{index}] {metadata_text} cite as <|{file_id}|>\n{chunk_text}\n"
    )

    results = [make_result(file_id="a", filename="a.pdf", score=0.9, text="first chunk")]
    items = file_search.build_content_items("my query", results, config)
    chunk_text = items[1].text

    # The customized (disabled-branch) template must not be used
    assert chunk_text == VectorStoresConfig().annotation_prompt_params.chunk_annotation_template.format(
        index=1, metadata_text="document_id: a, score: 0.9", file_id="a", chunk_text="first chunk"
    )


def test_build_citation_files_orders_by_score_and_falls_back_to_attributes():
    results = [
        make_result(
            file_id="", filename="", score=0.95, attributes={"document_id": "from-attrs", "filename": "attr.pdf"}
        ),
        make_result(file_id="b", filename="b.pdf", score=0.5),
        make_result(file_id="", score=0.1, filename=""),
    ]

    citations = file_search.build_citation_files(results)

    # Descending score order; attribute fallbacks applied; missing ids stay empty
    assert list(citations.items()) == [("from-attrs", "attr.pdf"), ("b", "b.pdf"), ("", "unknown")]


def test_build_metadata_shape():
    results = [
        make_result(file_id="a", filename="a.pdf", score=0.9, text="first chunk", attributes={"k": "v"}),
        make_result(file_id="b", filename="b.pdf", score=0.4, text="second chunk"),
    ]
    citation_files = file_search.build_citation_files(results)

    metadata = file_search.build_metadata(results, citation_files)

    assert metadata["document_ids"] == ["a", "b"]
    assert metadata["chunks"] == ["first chunk", "second chunk"]
    assert metadata["scores"] == [0.9, 0.4]
    assert metadata["attributes"] == [{"k": "v"}, {}]
    assert metadata["citation_files"] == {"a": "a.pdf", "b": "b.pdf"}
