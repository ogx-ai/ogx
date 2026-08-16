# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace

from ogx.providers.inline.file_processor.docling._metadata import extract_structural_metadata
from ogx.providers.utils.files.structural_metadata import structural_metadata_as_attributes
from ogx_api.vector_io import VectorStoreContent, VectorStoreSearchResponse


def test_extract_structural_metadata_from_native_docling_chunk():
    doc_chunk = SimpleNamespace(
        headings=["wrong location"],
        meta=SimpleNamespace(
            headings=["Introduction", "Architecture"],
            doc_items=[
                SimpleNamespace(prov=[SimpleNamespace(page_no=2), SimpleNamespace(page_no=1)]),
                SimpleNamespace(prov=[SimpleNamespace(page_no=2)]),
                SimpleNamespace(prov=None),
            ],
        ),
    )

    assert extract_structural_metadata(doc_chunk) == {
        "headings": "Introduction > Architecture",
        "page_numbers": "1, 2",
    }


def test_extract_structural_metadata_omits_empty_values():
    doc_chunk = SimpleNamespace(meta=SimpleNamespace(headings=None, doc_items=[]))

    assert extract_structural_metadata(doc_chunk) == {}


def test_extract_structural_metadata_preserves_legacy_top_level_headings():
    doc_chunk = SimpleNamespace(
        headings=["Legacy heading"],
        meta=SimpleNamespace(headings=None, doc_items=[]),
    )

    assert extract_structural_metadata(doc_chunk) == {
        "headings": ["Legacy heading"],
    }


def test_structural_metadata_attributes_keep_commas_inside_headings():
    assert structural_metadata_as_attributes(
        headings=["Safety, Security", "Database setup"],
        page_numbers=[4, 5],
    ) == {
        "headings": "Safety, Security > Database setup",
        "page_numbers": "4, 5",
    }


def test_structural_metadata_attributes_preserve_existing_strings():
    assert structural_metadata_as_attributes(
        headings="Database setup",
        page_numbers="4, 5",
    ) == {
        "headings": "Database setup",
        "page_numbers": "4, 5",
    }


def test_structural_metadata_is_valid_in_vector_store_search_attributes():
    metadata = structural_metadata_as_attributes(
        headings=["Installation", "Database setup"],
        page_numbers=[4, 5],
    )

    result = VectorStoreSearchResponse(
        file_id="file-123",
        filename="manual.pdf",
        score=1.0,
        attributes=metadata,
        content=[VectorStoreContent(type="text", text="Database setup instructions")],
    )

    assert result.attributes == {
        "headings": "Installation > Database setup",
        "page_numbers": "4, 5",
    }
