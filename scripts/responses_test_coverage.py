#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Responses API Integration Test Coverage Analyzer

The expected feature set is derived from the OpenAI API spec and the
ogx fastapi_routes.py files. Coverage detection uses AST analysis
of integration tests.

Usage:
    uv run python scripts/responses_test_coverage.py [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests" / "integration" / "responses"
OPENAI_SPEC = ROOT / "docs" / "static" / "openai-spec-2.3.0.yml"
AGENTS_ROUTES = ROOT / "src" / "ogx_api" / "responses" / "fastapi_routes.py"
CONVERSATIONS_ROUTES = ROOT / "src" / "ogx_api" / "conversations" / "fastapi_routes.py"


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _load_spec(path: Path) -> dict[str, Any]:
    content = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        return yaml.safe_load(content)
    return json.loads(content)


def _resolve_ref(ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    parts = ref.lstrip("#/").split("/")
    obj: Any = spec
    for p in parts:
        obj = obj[p]
    return obj


def _collect_properties(
    schema: dict[str, Any], spec: dict[str, Any], visited: set[str] | None = None
) -> dict[str, Any]:
    """Recursively collect all property names from a schema, resolving $ref."""
    if visited is None:
        visited = set()
    props: dict[str, Any] = {}
    if not isinstance(schema, dict):
        return props
    ref = schema.get("$ref")
    if ref:
        if ref in visited:
            return props
        visited.add(ref)
        return _collect_properties(_resolve_ref(ref, spec), spec, visited)
    if "properties" in schema:
        for name, val in schema["properties"].items():
            props[name] = val
    for key in ("allOf", "oneOf", "anyOf"):
        if key in schema:
            for sub in schema[key]:
                props.update(_collect_properties(sub, spec, visited))
    return props


def _get_type_value(schema: dict[str, Any], spec: dict[str, Any]) -> str | None:
    """Extract the 'type' discriminator value from a schema (e.g. 'function', 'response.created')."""
    # Direct properties
    tp = schema.get("properties", {}).get("type", {})
    if tp:
        return tp.get("enum", [None])[0] or tp.get("const")
    # Walk allOf
    for sub in schema.get("allOf", []):
        if "$ref" in sub:
            sub = _resolve_ref(sub["$ref"], spec)
        tp = sub.get("properties", {}).get("type", {})
        if tp:
            return tp.get("enum", [None])[0] or tp.get("const")
    return None


def _extract_oneof_types(ref: str, spec: dict[str, Any]) -> list[str]:
    """Extract all type discriminator values from a oneOf/anyOf union."""
    schema = _resolve_ref(ref, spec)
    types = []
    for key in ("oneOf", "anyOf"):
        for item in schema.get(key, []):
            if "$ref" in item:
                resolved = _resolve_ref(item["$ref"], spec)
                val = _get_type_value(resolved, spec)
                if val:
                    types.append(val)
    return types


# ---------------------------------------------------------------------------
# Route extraction from fastapi_routes.py
# ---------------------------------------------------------------------------


def _extract_routes_from_file(filepath: Path) -> list[tuple[str, str]]:
    """Parse a fastapi_routes.py and return (method, path) tuples."""
    source = filepath.read_text()
    tree = ast.parse(source)
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "router":
                method = node.func.attr
                if method in ("get", "post", "put", "delete", "patch"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        path = node.args[0].value
                        routes.append((method, path))
    return routes


def _parse_route(method: str, path: str) -> tuple[str, str, str]:
    """Convert a (method, path) to (feature_id, sdk_method, description)."""
    prefix = "crud" if path.startswith("/responses") else "conv"
    base = "responses" if path.startswith("/responses") else "conversations"
    segments = [s for s in path.split("/") if s and not s.startswith("{")]
    sub_segments = segments[1:]

    method_map = {"post": "create", "get": "list", "delete": "delete"}
    action = method_map.get(method, method)

    if not sub_segments:
        if method == "get" and "{" in path:
            action = "retrieve"
        fid = f"{prefix}.{action}"
        sdk = f"{base}.{action}"
    else:
        suffix = "_".join(sub_segments)
        sub = ".".join(sub_segments)
        if method == "delete":
            fid = f"{prefix}.delete_{suffix}"
            sdk = f"{base}.{sub}.delete"
        elif method == "get":
            fid = f"{prefix}.{suffix}"
            sdk = f"{base}.{sub}.list" if "{" not in sub_segments[-1] else f"{base}.{sub}"
        else:
            fid = f"{prefix}.{suffix}"
            sdk = f"{base}.{sub}"

    return fid, sdk, f"{method.upper()} {path}"


# Params to skip — always present or not meaningfully testable
_SKIP_PARAMS = {
    "prompt",  # internal/deprecated
    "user",  # identity param
    "stream_options",  # sub-option of stream
    "prompt_cache_retention",  # not yet supported
}


@dataclass
class Feature:
    """A testable feature of the Responses API."""

    id: str
    category: str
    description: str
    sdk_method: str = ""  # SDK method to match in tests (e.g. 'responses.create')
    property_names: list[str] = field(default_factory=list)
    covered: bool = False
    test_locations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature matrix builder — spec + routes driven
# ---------------------------------------------------------------------------


def build_feature_matrix(
    openai_spec_path: Path = OPENAI_SPEC,
    agents_routes_path: Path = AGENTS_ROUTES,
    conversations_routes_path: Path = CONVERSATIONS_ROUTES,
) -> list[Feature]:
    """Build feature list from the OpenAI API spec and fastapi route files."""
    spec = _load_spec(openai_spec_path)
    features: list[Feature] = []

    # --- Request parameters from POST /responses schema ---
    create_ref = spec["paths"]["/responses"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    req_props = _collect_properties({"$ref": create_ref}, spec)
    for name in sorted(req_props.keys()):
        if name in _SKIP_PARAMS:
            continue
        features.append(
            Feature(
                id=f"param.{name}",
                category="Request Parameters",
                description=f"{name} parameter",
                property_names=[name],
            )
        )

    # --- Tool types from Tool oneOf ---
    tool_types = _extract_oneof_types("#/components/schemas/Tool", spec)
    seen_types: set[str] = set()
    for type_val in tool_types:
        # Normalize versioned types like 'web_search_2025_08_26' -> 'web_search'
        normalized = type_val.split("_20")[0] if "_20" in type_val else type_val
        if normalized in seen_types:
            continue
        seen_types.add(normalized)
        features.append(
            Feature(
                id=f"tools.{normalized}",
                category="Tools",
                description=f"{normalized} tool",
                property_names=["tools"],
            )
        )
    # function_call_output (behavioral — multi-turn tool use)
    features.append(
        Feature(
            id="tools.function_call_output",
            category="Tools",
            description="function_call_output in multi-turn",
            property_names=["output"],
        )
    )

    # --- Structured output sub-features ---
    for fmt in ("json_schema", "json_object"):
        features.append(
            Feature(
                id=f"text.{fmt}",
                category="Structured Output",
                description=f"text format {fmt}",
                property_names=["text"],
            )
        )

    # --- Streaming events from ResponseStreamEvent oneOf ---
    event_types = _extract_oneof_types("#/components/schemas/ResponseStreamEvent", spec)
    for event_type in event_types:
        # event_type is like 'response.created', 'response.output_text.delta'
        short = event_type.replace("response.", "", 1)
        features.append(
            Feature(
                id=f"stream.{short}",
                category="Streaming Events",
                description=f"{event_type} event",
            )
        )

    # --- CRUD and Conversation endpoints from fastapi_routes.py ---
    for routes_path in (agents_routes_path, conversations_routes_path):
        if not routes_path.exists():
            continue
        for method, path in _extract_routes_from_file(routes_path):
            fid, sdk_method, desc = _parse_route(method, path)
            category = "CRUD Operations" if path.startswith("/responses") else "Conversations"
            features.append(Feature(id=fid, category=category, description=desc, sdk_method=sdk_method))

    # conversation= param in responses.create
    features.append(
        Feature(
            id="conv.with_response",
            category="Conversations",
            description="conversation= param in responses.create",
        )
    )

    return features


# ---------------------------------------------------------------------------
# AST-based test analysis
# ---------------------------------------------------------------------------


def _get_call_chain(node: ast.Call) -> str | None:
    """Reconstruct dotted call chain like 'openai_client.responses.create'."""
    parts: list[str] = []
    obj = node.func
    while isinstance(obj, ast.Attribute):
        parts.append(obj.attr)
        obj = obj.value
    if isinstance(obj, ast.Name):
        parts.append(obj.id)
    else:
        return None
    parts.reverse()
    return ".".join(parts)


def _is_openai_call(chain: str) -> bool:
    return any(chain.startswith(c) for c in ("openai_client.", "alice_client.", "bob_client.", "self."))


def _strip_client_prefix(chain: str) -> str:
    """'openai_client.responses.create' -> 'responses.create'"""
    return chain.split(".", 1)[1] if "." in chain else chain


@dataclass
class TestEvidence:
    """Evidence of what a test exercises, extracted via AST."""

    params: set[str] = field(default_factory=set)
    tool_types: set[str] = field(default_factory=set)
    text_formats: set[str] = field(default_factory=set)
    api_methods: set[str] = field(default_factory=set)
    stream_events: set[str] = field(default_factory=set)
    has_function_call_output: bool = False


def _analyze_test_ast(func_node: ast.AST) -> TestEvidence:
    """Walk a test function's AST and extract coverage evidence."""
    ev = TestEvidence()

    for node in ast.walk(func_node):
        # --- API calls and their keyword args ---
        if isinstance(node, ast.Call):
            chain = _get_call_chain(node)
            if chain and _is_openai_call(chain):
                method = _strip_client_prefix(chain)
                ev.api_methods.add(method)
                for kw in node.keywords:
                    if kw.arg:
                        ev.params.add(kw.arg)

        # --- Dict literals: detect tool types and text formats ---
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=False):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant) and isinstance(v.value, str):
                    if k.value == "type":
                        # Normalize versioned types
                        val = v.value.split("_20")[0] if "_20" in v.value else v.value
                        ev.tool_types.add(val)
                        if val in ("json_schema", "json_object"):
                            ev.text_formats.add(val)
                        if val == "function_call_output":
                            ev.has_function_call_output = True

        # --- String constants: detect stream event types ---
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val.startswith("response.") and len(val) > len("response."):
                ev.stream_events.add(val)

    return ev


# ---------------------------------------------------------------------------
# Test scanning and matching
# ---------------------------------------------------------------------------


def _extract_openai_test_functions(filepath: Path) -> list[tuple[str, ast.AST]]:
    """Return (location_key, func_node) for tests that use openai_client."""
    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"  WARNING: Could not parse {filepath}", file=sys.stderr)
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_"):
            arg_names = [arg.arg for arg in node.args.args]
            if any(a in arg_names for a in ("openai_client", "alice_client", "bob_client")):
                location = f"{filepath.relative_to(ROOT)}:{node.lineno}::{node.name}"
                results.append((location, node))
    return results


def _scan_streaming_helpers(test_dir: Path) -> set[str]:
    """Extract stream event types from streaming_assertions.py."""
    helpers = test_dir / "streaming_assertions.py"
    if not helpers.exists():
        return set()
    source = helpers.read_text()
    tree = ast.parse(source)
    events: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val.startswith("response.") and len(val) > len("response."):
                events.add(val)
    return events


def _match_evidence(features: list[Feature], evidence_map: dict[str, TestEvidence], helper_events: set[str]) -> None:
    """Match accumulated test evidence against features."""
    # Aggregate all evidence into per-kind lookup dicts
    agg: dict[str, dict[str, list[str]]] = {}
    fn_call_output_locs: list[str] = []
    for loc, ev in evidence_map.items():
        for kind, values in [
            ("params", ev.params),
            ("tool_types", ev.tool_types),
            ("text_formats", ev.text_formats),
            ("api_methods", ev.api_methods),
            ("stream_events", ev.stream_events),
        ]:
            for v in values:
                agg.setdefault(kind, {}).setdefault(v, []).append(loc)
        if ev.has_function_call_output:
            fn_call_output_locs.append(loc)

    for e in helper_events:
        agg.setdefault("stream_events", {}).setdefault(e, []).append("streaming_assertions.py")

    for feat in features:
        locs: list[str] = []

        if feat.id.startswith("param."):
            locs = agg.get("params", {}).get(feat.id[len("param.") :], [])

        elif feat.id == "tools.function_call_output":
            locs = fn_call_output_locs

        elif feat.id.startswith("tools."):
            locs = agg.get("tool_types", {}).get(feat.id[len("tools.") :], [])

        elif feat.id.startswith("text."):
            locs = agg.get("text_formats", {}).get(feat.id[len("text.") :], [])

        elif feat.id.startswith("stream."):
            event_type = "response." + feat.id[len("stream.") :]
            locs = agg.get("stream_events", {}).get(event_type, [])

        elif feat.id == "conv.with_response":
            locs = agg.get("params", {}).get("conversation", [])

        elif feat.sdk_method:
            locs = agg.get("api_methods", {}).get(feat.sdk_method, [])

        if locs:
            feat.covered = True
            feat.test_locations = list(dict.fromkeys(locs))


def run_coverage(test_dir: Path = TESTS_DIR, spec_path: Path = OPENAI_SPEC) -> list[Feature]:
    """Build features from spec, scan tests via AST, and match coverage."""
    features = build_feature_matrix(spec_path)

    evidence_map: dict[str, TestEvidence] = {}
    for filepath in sorted(test_dir.glob("test_*.py")):
        for location, func_node in _extract_openai_test_functions(filepath):
            evidence_map[location] = _analyze_test_ast(func_node)

    helper_events = _scan_streaming_helpers(test_dir)
    _match_evidence(features, evidence_map, helper_events)

    return features


def get_tested_property_names(features: list[Feature] | None = None) -> set[str]:
    """Return the set of OpenAI spec property names that have integration test coverage.

    Includes both request params from covered features and response attributes
    accessed in tests (e.g. response.usage, response.error).
    """
    if features is None:
        features = run_coverage()

    tested = set()
    for feat in features:
        if feat.covered:
            tested.update(feat.property_names)

    # Also scan for response attributes accessed in tests (for conformance annotation)
    _response_var_names = {"response", "response1", "response2", "retrieved"}
    for filepath in sorted(TESTS_DIR.glob("test_*.py")):
        for _, func_node in _extract_openai_test_functions(filepath):
            for node in ast.walk(func_node):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id in _response_var_names:
                        tested.add(node.attr)
    return tested


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "Request Parameters",
    "Tools",
    "Structured Output",
    "Streaming Events",
    "CRUD Operations",
    "Conversations",
]


def print_report(features: list[Feature], verbose: bool = False) -> float:
    """Print coverage report and return overall score."""
    categories: dict[str, list[Feature]] = {}
    for feat in features:
        categories.setdefault(feat.category, []).append(feat)

    total = len(features)
    covered = sum(1 for feat in features if feat.covered)
    score = (covered / total * 100) if total > 0 else 0

    print("=" * 72)
    print("  Responses API — OpenAI Client Integration Test Coverage")
    print("=" * 72)
    print()
    print(f"  Overall Score: {score:.1f}% ({covered}/{total} features covered)")
    print()

    print(f"{'Category':<25} {'Covered':>8} {'Total':>8} {'Score':>8}")
    print("-" * 55)
    for cat_name in CATEGORY_ORDER:
        cat_features = categories.get(cat_name, [])
        if not cat_features:
            continue
        cat_covered = sum(1 for feat in cat_features if feat.covered)
        cat_total = len(cat_features)
        cat_score = (cat_covered / cat_total * 100) if cat_total > 0 else 0
        print(f"{cat_name:<25} {cat_covered:>8} {cat_total:>8} {cat_score:>7.1f}%")

    print()

    gaps = [feat for feat in features if not feat.covered]
    if gaps:
        print(f"GAPS ({len(gaps)} features missing coverage):")
        print()
        current_cat = None
        for feat in gaps:
            if feat.category != current_cat:
                current_cat = feat.category
                print(f"  [{current_cat}]")
            print(f"    - {feat.id}: {feat.description}")
        print()

    if verbose:
        print("COVERED FEATURES:")
        print()
        current_cat = None
        for feat in features:
            if not feat.covered:
                continue
            if feat.category != current_cat:
                current_cat = feat.category
                print(f"  [{current_cat}]")
            locs = ", ".join(feat.test_locations[:3])
            if len(feat.test_locations) > 3:
                locs += f" (+{len(feat.test_locations) - 3} more)"
            print(f"    + {feat.id}: {feat.description}")
            print(f"      {locs}")
        print()

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Responses API test coverage analyzer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show covered features with test locations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not TESTS_DIR.exists():
        print(f"ERROR: Test directory not found: {TESTS_DIR}", file=sys.stderr)
        sys.exit(1)

    features = run_coverage()

    if args.json:
        data = {
            "score": round(sum(1 for f in features if f.covered) / len(features) * 100, 1),
            "total": len(features),
            "covered": sum(1 for f in features if f.covered),
            "tested_properties": sorted(get_tested_property_names(features)),
            "gaps": [
                {"id": f.id, "category": f.category, "description": f.description} for f in features if not f.covered
            ],
            "covered_features": [
                {
                    "id": f.id,
                    "category": f.category,
                    "description": f.description,
                    "test_locations": f.test_locations,
                }
                for f in features
                if f.covered
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"\nScanned tests from {TESTS_DIR.relative_to(ROOT)}/\n")
        print_report(features, verbose=args.verbose)


if __name__ == "__main__":
    main()
