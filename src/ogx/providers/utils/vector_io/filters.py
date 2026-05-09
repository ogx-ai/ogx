# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Re-export filter types from the API package for implementation convenience.

This allows implementation code to import filters from the utils package
while the actual definitions remain in the API package for proper OpenAPI generation.
"""

from typing import Any

# Re-export filter types from the API package
from ogx_api import ComparisonFilter, CompoundFilter, Filter
from ogx_api.filters import ALL_FILTER_TYPES, COMPARISON_FILTER_TYPES, COMPOUND_FILTER_TYPES


def parse_filter(filter_data: Any) -> ComparisonFilter | CompoundFilter:
    """Recursively parse filter data into typed Filter objects.

    Converts dictionary-based filter data into proper typed Filter objects
    (ComparisonFilter or CompoundFilter), supporting arbitrary nesting and
    pre-typed filter objects passed through unchanged.

    Args:
        filter_data: Filter data as dict, ComparisonFilter, or CompoundFilter

    Returns:
        Typed filter object (ComparisonFilter or CompoundFilter)

    Raises:
        ValueError: If filter data is invalid or has unsupported structure
    """
    # Handle pre-typed filter objects — pass through unchanged
    if isinstance(filter_data, ComparisonFilter | CompoundFilter):
        return filter_data

    if not isinstance(filter_data, dict):
        raise ValueError("Filter must be a dict or typed Filter object")

    filter_type = filter_data.get("type")
    if not filter_type:
        raise ValueError("Filter must have a 'type' field")

    if filter_type in COMPARISON_FILTER_TYPES:
        if "key" not in filter_data or "value" not in filter_data:
            raise ValueError(f"Comparison filter '{filter_type}' must have 'key' and 'value' fields")
        return ComparisonFilter(**filter_data)

    if filter_type in COMPOUND_FILTER_TYPES:
        sub_filters_data = filter_data.get("filters", [])
        if not isinstance(sub_filters_data, list):
            raise ValueError(f"Compound filter '{filter_type}' must have a 'filters' list")
        parsed_sub_filters = [parse_filter(sf) for sf in sub_filters_data]
        return CompoundFilter(type=filter_type, filters=parsed_sub_filters)

    supported_types = ", ".join(sorted(ALL_FILTER_TYPES))
    raise ValueError(f"Invalid filter type: '{filter_type}'. Supported types: {supported_types}")


__all__ = ["ComparisonFilter", "CompoundFilter", "Filter", "parse_filter"]
