# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any


def _attribute_value(value: Any, separator: str) -> str:
    if isinstance(value, list | tuple):
        return separator.join(str(item) for item in value if str(item))
    return str(value) if value is not None else ""


def structural_metadata_as_attributes(*, headings: Any = None, page_numbers: Any = None) -> dict[str, str]:
    """Convert structural chunk metadata to scalar vector-store attributes."""
    metadata: dict[str, str] = {}

    headings_value = _attribute_value(headings, " > ")
    if headings_value:
        metadata["headings"] = headings_value

    page_numbers_value = _attribute_value(page_numbers, ", ")
    if page_numbers_value:
        metadata["page_numbers"] = page_numbers_value

    return metadata
