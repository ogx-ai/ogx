# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace

from ogx.providers.inline.file_processor.docling._metadata import extract_structural_metadata


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
        "headings": ["Introduction", "Architecture"],
        "page_numbers": [1, 2],
    }


def test_extract_structural_metadata_omits_empty_values():
    doc_chunk = SimpleNamespace(meta=SimpleNamespace(headings=None, doc_items=[]))

    assert extract_structural_metadata(doc_chunk) == {}
