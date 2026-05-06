# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAI-compatible filter types for vector store search operations.

These filter types allow filtering search results based on metadata fields,
supporting both comparison operations (eq, ne, gt, etc.) and compound
operations (and, or) for complex filtering logic.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .schema_utils import json_schema_type

# Constants for filter type validation (using sets for O(1) membership testing)
COMPARISON_FILTER_TYPES = frozenset(["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"])
COMPOUND_FILTER_TYPES = frozenset(["and", "or"])
ALL_FILTER_TYPES = COMPARISON_FILTER_TYPES | COMPOUND_FILTER_TYPES


@json_schema_type
class ComparisonFilter(BaseModel):
    """A filter that compares a metadata field against a value.

    :param type: The comparison operator to apply
    :param key: The metadata field name to filter on
    :param value: The value to compare against
    """

    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
    key: str
    value: Any


@json_schema_type
class CompoundFilter(BaseModel):
    """A filter that combines multiple filters with a logical operator.

    :param type: The logical operator ("and" requires all filters match, "or" requires any filter matches)
    :param filters: The list of filters to combine
    """

    type: Literal["and", "or"]
    # Temporarily use Any to break recursion, will be updated after Filter definition
    filters: list[Any]


# Define the type for internal use
FilterUnion = ComparisonFilter | CompoundFilter

# Now update CompoundFilter with proper type
CompoundFilter.model_fields["filters"].annotation = list[FilterUnion]
CompoundFilter.model_rebuild()

# Create the annotated version for API use
Filter = Annotated[
    FilterUnion,
    Field(discriminator="type"),
]
