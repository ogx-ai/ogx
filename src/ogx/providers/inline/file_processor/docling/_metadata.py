# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any


def extract_structural_metadata(doc_chunk: Any) -> dict[str, Any]:
    chunk_meta = getattr(doc_chunk, "meta", None)
    if chunk_meta is None:
        return {}

    metadata: dict[str, Any] = {}
    headings = getattr(chunk_meta, "headings", None)
    if headings:
        metadata["headings"] = headings

    page_numbers = {
        page_number
        for doc_item in getattr(chunk_meta, "doc_items", [])
        for provenance in (getattr(doc_item, "prov", None) or [])
        if (page_number := getattr(provenance, "page_no", None)) is not None
    }
    if page_numbers:
        metadata["page_numbers"] = sorted(page_numbers)

    return metadata
