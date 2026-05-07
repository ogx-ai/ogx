# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Contract tests for the ogx_api public surface.

These tests lock in the two-namespace split documented in
`docs/docs/concepts/apis/api_leveling.mdx`:

- `ogx_api.provider.*` exposes the Provider SDK surface (protocol classes,
  ProviderSpec family, schema utilities, Api enum, Resource base classes,
  version constants, validators, and shared resource/value types).
- `ogx_api.types.*` exposes the API datatype surface (request/response
  Pydantic models, content types, errors, value types).

Both namespaces remain importable from the top level for backwards
compatibility.
"""

import ogx_api
import ogx_api.provider as provider
import ogx_api.types as types


def test_provider_and_types_are_disjoint() -> None:
    """A symbol should belong to one surface only.

    Shared resource/value types (Model, Shield, ToolGroup, VectorStore) are
    canonically in `types` and re-exported from `provider` for convenience —
    these are the only allowed overlaps.
    """
    allowed_overlap = {"Model", "Shield", "ToolGroup", "VectorStore"}
    overlap = (set(provider.__all__) & set(types.__all__)) - allowed_overlap
    assert not overlap, f"Symbols in both namespaces: {sorted(overlap)}"


def test_provider_symbols_resolve_to_same_object_at_top_level() -> None:
    """Provider SDK symbols must remain importable from `ogx_api` directly."""
    for name in provider.__all__:
        assert hasattr(ogx_api, name), f"{name} not exported from ogx_api top level"
        assert getattr(ogx_api, name) is getattr(provider, name), f"{name} differs between ogx_api and ogx_api.provider"


def test_types_symbols_resolve_to_same_object_at_top_level() -> None:
    """API datatype symbols must remain importable from `ogx_api` directly."""
    for name in types.__all__:
        assert hasattr(ogx_api, name), f"{name} not exported from ogx_api top level"
        assert getattr(ogx_api, name) is getattr(types, name), f"{name} differs between ogx_api and ogx_api.types"


def test_provider_surface_contains_core_symbols() -> None:
    """Spot-check load-bearing Provider SDK symbols."""
    required = {
        "Api",
        "ProviderSpec",
        "InlineProviderSpec",
        "RemoteProviderSpec",
        "Inference",
        "Responses",
        "VectorIO",
        "Safety",
        "Resource",
        "ResourceType",
        "OGX_API_V1",
        "json_schema_type",
        "register_schema",
    }
    missing = required - set(provider.__all__)
    assert not missing, f"Provider SDK missing required symbols: {sorted(missing)}"


def test_types_surface_contains_core_symbols() -> None:
    """Spot-check load-bearing API datatype symbols."""
    required = {
        "OpenAIResponseObject",
        "ChatCompletionMessage",
        "OpenAIChatCompletion",
        "ListModelsResponse",
        "Filter",
        "Model",
        "Shield",
        "VectorStore",
    }
    missing = required - set(types.__all__)
    assert not missing, f"Types surface missing required symbols: {sorted(missing)}"
