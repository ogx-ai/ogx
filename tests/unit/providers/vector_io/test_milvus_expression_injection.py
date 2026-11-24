# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

r"""
Tests for Milvus filter expression construction (expression-injection prevention).

Milvus filter expressions are built by string interpolation. Values are
formatted as escaped expression literals, and range operators require numeric
values, so a filter value cannot break out of its literal and inject
expression syntax.

Run:  uv run pytest tests/unit/providers/vector_io/test_milvus_expression_injection.py -v -s
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

if "pymilvus" not in sys.modules:
    pymilvus = ModuleType("pymilvus")
    pymilvus.AnnSearchRequest = object
    pymilvus.DataType = SimpleNamespace(
        VARCHAR="VARCHAR",
        FLOAT_VECTOR="FLOAT_VECTOR",
        JSON="JSON",
        SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
    )
    pymilvus.Function = object
    pymilvus.FunctionType = SimpleNamespace(BM25="BM25")
    pymilvus.AsyncMilvusClient = object
    pymilvus.RRFRanker = object
    pymilvus.WeightedRanker = object
    sys.modules["pymilvus"] = pymilvus

from ogx.providers.remote.vector_io.milvus.milvus import MilvusIndex
from ogx_api.filters import ComparisonFilter, CompoundFilter


class _FakeClient:
    pass


@pytest.fixture
def index():
    vector_store = SimpleNamespace(identifier="test-store", embedding_dimension=768)
    return MilvusIndex(client=_FakeClient(), vector_store=vector_store)


@pytest.mark.parametrize(
    ("filter_obj", "expected"),
    [
        (
            ComparisonFilter(type="eq", key="author", value="alice"),
            "chunk_content['metadata']['author'] == \"alice\"",
        ),
        (
            ComparisonFilter(type="ne", key="author", value="bob"),
            "chunk_content['metadata']['author'] != \"bob\"",
        ),
        (ComparisonFilter(type="eq", key="active", value=True), "chunk_content['metadata']['active'] == true"),
        (ComparisonFilter(type="eq", key="active", value=False), "chunk_content['metadata']['active'] == false"),
        (ComparisonFilter(type="eq", key="page", value=5), "chunk_content['metadata']['page'] == 5"),
        (ComparisonFilter(type="eq", key="score", value=2.5), "chunk_content['metadata']['score'] == 2.5"),
        (ComparisonFilter(type="gt", key="page", value=10), "chunk_content['metadata']['page'] > 10"),
        (ComparisonFilter(type="gte", key="page", value=10), "chunk_content['metadata']['page'] >= 10"),
        (ComparisonFilter(type="lt", key="page", value=10), "chunk_content['metadata']['page'] < 10"),
        (ComparisonFilter(type="lte", key="page", value=10), "chunk_content['metadata']['page'] <= 10"),
        (
            ComparisonFilter(type="in", key="tags", value=["a", "b"]),
            "chunk_content['metadata']['tags'] in [\"a\", \"b\"]",
        ),
        (
            ComparisonFilter(type="nin", key="tags", value=["a"]),
            "chunk_content['metadata']['tags'] not in [\"a\"]",
        ),
    ],
)
def test_valid_filters_translate_as_before(index, filter_obj, expected):
    """Legitimate filters must produce exactly the expressions they always did."""
    assert index._translate_comparison_filter(filter_obj) == expected


@pytest.mark.parametrize(
    ("value", "expected_literal"),
    [
        ('a" or 1==1 or "b', '"a\\" or 1==1 or \\"b"'),
        ("back\\slash", '"back\\\\slash"'),
        ("' OR 1=1 --", '"\' OR 1=1 --"'),
    ],
)
def test_eq_string_values_are_escaped_literals(index, value, expected_literal):
    """A string value containing quotes or backslashes stays inside one literal."""
    expr = index._translate_comparison_filter(ComparisonFilter(type="eq", key="k", value=value))
    assert expr == f"chunk_content['metadata']['k'] == {expected_literal}"


@pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
@pytest.mark.parametrize("value", [True, False, None, "1", "0 or 1 == 1", [1], {"a": 1}])
def test_range_ops_reject_non_numeric_values(index, op, value):
    """Range operators must refuse anything that is not a plain number."""
    with pytest.raises(ValueError, match="numeric value required"):
        index._translate_comparison_filter(ComparisonFilter(type=op, key="n", value=value))


def test_in_nin_escape_elements(index):
    expr = index._translate_comparison_filter(ComparisonFilter(type="in", key="tags", value=['a"b', "c\\d"]))
    assert expr == 'chunk_content[\'metadata\'][\'tags\'] in ["a\\"b", "c\\\\d"]'


@pytest.mark.parametrize("element", [None, {"a": 1}, [1], b"bytes"])
def test_in_rejects_non_scalar_elements(index, element):
    with pytest.raises(ValueError, match="Failed to format filter value"):
        index._translate_comparison_filter(ComparisonFilter(type="in", key="k", value=[element]))


@pytest.mark.parametrize("value", [None, {"a": 1}, [1], b"bytes"])
def test_eq_rejects_unsupported_types(index, value):
    with pytest.raises(ValueError, match="Failed to format filter value"):
        index._translate_comparison_filter(ComparisonFilter(type="eq", key="k", value=value))


def test_in_requires_list(index):
    with pytest.raises(ValueError, match="requires a list value"):
        index._translate_comparison_filter(ComparisonFilter(type="in", key="k", value="a"))


def test_compound_filter_escapes_all_clauses(index):
    f = CompoundFilter(
        type="and",
        filters=[
            ComparisonFilter(type="eq", key="a", value='x"y'),
            ComparisonFilter(type="gt", key="n", value=1),
        ],
    )
    assert (
        index._translate_compound_filter(f)
        == "(chunk_content['metadata']['a'] == \"x\\\"y\") and (chunk_content['metadata']['n'] > 1)"
    )
