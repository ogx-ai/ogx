#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Google Interactions API Coverage Analyzer

Compares OGX's Interactions implementation against Google's official
OpenAPI spec and generates a coverage report showing:
- Which endpoints are implemented
- Which request/response properties are supported
- Which content types, tools, and streaming events are covered
- A conformance score

Usage:
    python scripts/google_interactions_coverage.py [--update] [--generate-docs] [--check-regression] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SPEC = REPO_ROOT / "docs" / "static" / "google-interactions-spec.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "static" / "google-interactions-coverage.json"


def _load_spec(path: Path) -> dict[str, Any]:
    """Load the Google Interactions OpenAPI spec."""
    return json.loads(path.read_text())


def _resolve_ref(ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref pointer in the spec."""
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    resolved = spec
    for part in parts:
        resolved = resolved.get(part, {})
    return resolved if isinstance(resolved, dict) else {}


def _get_schema_properties(
    schema: dict[str, Any],
    spec: dict[str, Any],
    visited: set[str] | None = None,
) -> dict[str, Any]:
    """Extract all direct property names and their schemas from a schema, resolving refs."""
    if visited is None:
        visited = set()

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in visited:
            return {}
        visited.add(ref)
        return _get_schema_properties(_resolve_ref(ref, spec), spec, visited)

    props = {}
    if "properties" in schema:
        props.update(schema["properties"])

    for key in ("allOf", "oneOf", "anyOf"):
        if key in schema:
            for sub in schema[key]:
                props.update(_get_schema_properties(sub, spec, visited))

    return props


def _collect_content_types(spec: dict[str, Any]) -> list[str]:
    """Extract all content type names from the Content oneOf schema."""
    content_schema = spec.get("components", {}).get("schemas", {}).get("Content", {})
    types = []
    for variant in content_schema.get("oneOf", []):
        if "$ref" in variant:
            name = variant["$ref"].split("/")[-1]
            types.append(name)
    return sorted(types)


def _collect_tool_types(spec: dict[str, Any]) -> list[str]:
    """Extract all tool type names from the Tool oneOf schema."""
    tool_schema = spec.get("components", {}).get("schemas", {}).get("Tool", {})
    types = []
    for variant in tool_schema.get("oneOf", []):
        if "$ref" in variant:
            name = variant["$ref"].split("/")[-1]
            types.append(name)
    return sorted(types)


def _collect_streaming_events(spec: dict[str, Any]) -> list[str]:
    """Extract all streaming event type names from InteractionSseEvent."""
    sse_schema = spec.get("components", {}).get("schemas", {}).get("InteractionSseEvent", {})
    events = []
    for variant in sse_schema.get("oneOf", []):
        if "$ref" in variant:
            name = variant["$ref"].split("/")[-1]
            events.append(name)
    return sorted(events)


# ──────────────────────────────────────────────────────────────────────
# OGX implementation model
# ──────────────────────────────────────────────────────────────────────

# What OGX currently implements, derived from the Pydantic models
# in src/ogx_api/interactions/models.py and the FastAPI routes.

IMPLEMENTED_ENDPOINTS = {
    "POST /{api_version}/interactions": True,
    "GET /{api_version}/interactions/{id}": False,
    "DELETE /{api_version}/interactions/{id}": False,
    "POST /{api_version}/interactions/{id}/cancel": False,
}

# Request properties we support (from GoogleCreateInteractionRequest)
IMPLEMENTED_REQUEST_PROPS = {
    "model",
    "input",
    "system_instruction",
    "generation_config",
    "stream",
    "response_modalities",
}

# Response properties we support (from GoogleInteractionResponse)
IMPLEMENTED_RESPONSE_PROPS = {
    "id",
    "created",
    "status",
    "updated",
    "model",
    "outputs",
    "role",
    "usage",
    "object",
}

# GenerationConfig properties we support
IMPLEMENTED_GENERATION_CONFIG_PROPS = {
    "temperature",
    "top_k",
    "top_p",
    "max_output_tokens",
}

# Usage properties we support
IMPLEMENTED_USAGE_PROPS = {
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
}

# Content types we support
IMPLEMENTED_CONTENT_TYPES = {
    "TextContent",
}

# Tool types we support
IMPLEMENTED_TOOL_TYPES: set[str] = set()

# Streaming events we support
IMPLEMENTED_STREAMING_EVENTS = {
    "InteractionStartEvent",
    "ContentStart",
    "ContentDelta",
    "ContentStop",
    "InteractionCompleteEvent",
}


def _compare_properties(
    section_name: str,
    google_props: dict[str, Any],
    implemented: set[str],
) -> dict[str, Any]:
    """Compare a set of Google spec properties against what we implement."""
    google_names = set(google_props.keys())
    supported = google_names & implemented
    missing = sorted(google_names - implemented)
    extra = sorted(implemented - google_names)

    total = len(google_names) if google_names else 1
    score = round(len(supported) / total * 100, 1)

    result: dict[str, Any] = {
        "section": section_name,
        "google_total": len(google_names),
        "implemented": len(supported),
        "missing_count": len(missing),
        "score": score,
        "supported": sorted(supported),
        "missing": missing,
    }
    if extra:
        result["extra_in_ogx"] = extra
    return result


def _compare_list(
    section_name: str,
    google_items: list[str],
    implemented: set[str],
) -> dict[str, Any]:
    """Compare a list of items from Google spec against what we implement."""
    google_set = set(google_items)
    supported = google_set & implemented
    missing = sorted(google_set - implemented)

    total = len(google_items) if google_items else 1
    score = round(len(supported) / total * 100, 1)

    return {
        "section": section_name,
        "google_total": len(google_items),
        "implemented": len(supported),
        "missing_count": len(missing),
        "score": score,
        "supported": sorted(supported),
        "missing": missing,
    }


def calculate_coverage(spec_path: Path) -> dict[str, Any]:
    """Calculate coverage of Google Interactions API."""
    spec = _load_spec(spec_path)
    schemas = spec.get("components", {}).get("schemas", {})

    sections: list[dict[str, Any]] = []

    # 1. Endpoints
    google_endpoints = []
    for path, path_item in spec.get("paths", {}).items():
        for method, op in path_item.items():
            if isinstance(op, dict) and not op.get("x-internal"):
                google_endpoints.append(f"{method.upper()} {path}")

    ep_google = set(google_endpoints)
    ep_impl = {ep for ep, supported in IMPLEMENTED_ENDPOINTS.items() if supported}
    ep_missing = sorted(ep_google - ep_impl)

    ep_total = len(ep_google) if ep_google else 1
    ep_section = {
        "section": "Endpoints",
        "google_total": len(ep_google),
        "implemented": len(ep_impl),
        "missing_count": len(ep_missing),
        "score": round(len(ep_impl) / ep_total * 100, 1),
        "supported": sorted(ep_impl),
        "missing": ep_missing,
    }
    sections.append(ep_section)

    # 2. Request properties (CreateModelInteractionParams)
    request_schema = schemas.get("CreateModelInteractionParams", {})
    request_props = _get_schema_properties(request_schema, spec)
    input_props = {k: v for k, v in request_props.items() if not (isinstance(v, dict) and v.get("readOnly"))}
    sections.append(_compare_properties("Request Properties", input_props, IMPLEMENTED_REQUEST_PROPS))

    # 3. Response properties (Interaction schema)
    response_schema = schemas.get("Interaction", {})
    response_props = _get_schema_properties(response_schema, spec)
    output_props = {
        k: v
        for k, v in response_props.items()
        if isinstance(v, dict) and (v.get("readOnly") or k in ("model", "agent"))
    }
    sections.append(_compare_properties("Response Properties", output_props, IMPLEMENTED_RESPONSE_PROPS))

    # 4. GenerationConfig
    gen_config_schema = schemas.get("GenerationConfig", {})
    gen_config_props = _get_schema_properties(gen_config_schema, spec)
    sections.append(_compare_properties("GenerationConfig", gen_config_props, IMPLEMENTED_GENERATION_CONFIG_PROPS))

    # 5. Usage
    usage_schema = schemas.get("Usage", {})
    usage_props = _get_schema_properties(usage_schema, spec)
    sections.append(_compare_properties("Usage", usage_props, IMPLEMENTED_USAGE_PROPS))

    # 6. Content types
    content_types = _collect_content_types(spec)
    sections.append(_compare_list("Content Types", content_types, IMPLEMENTED_CONTENT_TYPES))

    # 7. Tool types
    tool_types = _collect_tool_types(spec)
    sections.append(_compare_list("Tool Types", tool_types, IMPLEMENTED_TOOL_TYPES))

    # 8. Streaming events
    streaming_events = _collect_streaming_events(spec)
    sections.append(_compare_list("Streaming Events", streaming_events, IMPLEMENTED_STREAMING_EVENTS))

    # Overall score: weighted by google_total per section
    total_google = sum(s["google_total"] for s in sections)
    total_impl = sum(s["implemented"] for s in sections)
    overall_score = round(total_impl / total_google * 100, 1) if total_google else 0.0

    return {
        "google_spec": str(spec_path.relative_to(REPO_ROOT)),
        "spec_version": spec.get("info", {}).get("version", "unknown"),
        "summary": {
            "overall_score": overall_score,
            "total_google_items": total_google,
            "total_implemented": total_impl,
            "total_missing": total_google - total_impl,
        },
        "sections": sections,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    summary = report["summary"]

    print()
    print("=" * 65)
    print("Google Interactions API Coverage Report")
    print(f"Spec version: {report['spec_version']}")
    print("=" * 65)
    print()
    print(f"Overall Coverage: {summary['overall_score']}%")
    print(f"  Implemented: {summary['total_implemented']}/{summary['total_google_items']}")
    print(f"  Missing:     {summary['total_missing']}")
    print()

    print(f"{'Section':<25} {'Score':>6}  {'Impl':>5} / {'Total':>5}  {'Missing':>7}")
    print("-" * 65)

    for section in report["sections"]:
        name = section["section"]
        score = section["score"]
        impl = section["implemented"]
        total = section["google_total"]
        missing = section["missing_count"]
        print(f"{name:<25} {score:>5.1f}%  {impl:>5} / {total:>5}  {missing:>7}")

    print()

    for section in report["sections"]:
        if not section["missing"]:
            continue

        print(f"--- {section['section']} ---")
        print(f"  Missing ({section['missing_count']}):")
        for item in section["missing"]:
            print(f"    - {item}")
        if section.get("extra_in_ogx"):
            print("  Extra in OGX:")
            for item in section["extra_in_ogx"]:
                print(f"    + {item}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Google Interactions API coverage analyzer")
    parser.add_argument(
        "--google-spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="Path to Google Interactions OpenAPI spec",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for coverage JSON",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the coverage file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors",
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Also generate documentation from coverage report",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Fail if coverage score decreases compared to existing report",
    )

    args = parser.parse_args()

    if not args.google_spec.exists():
        print(f"Error: spec not found at {args.google_spec}")
        print(
            "Download it with: curl -o docs/static/google-interactions-spec.json"
            " https://ai.google.dev/api/interactions.openapi.json"
        )
        sys.exit(1)

    # Load existing coverage for regression check
    previous_score: float | None = None
    if args.check_regression and args.output.exists():
        try:
            with open(args.output) as f:
                previous_report = json.load(f)
                previous_score = previous_report.get("summary", {}).get("overall_score")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error: could not load previous report: {e}")
            sys.exit(1)

    report = calculate_coverage(args.google_spec)
    new_score = report["summary"]["overall_score"]

    # Check for coverage regression
    if args.check_regression and previous_score is not None:
        if new_score < previous_score:
            print(f"Coverage regression detected: {previous_score}% -> {new_score}%")
            print(f"Coverage decreased by {previous_score - new_score:.1f} percentage points")
            print()
            print("To fix this, ensure your changes don't reduce Google Interactions API coverage.")
            print("If this is intentional, update the coverage baseline with:")
            print(f"  python {__file__} --update --generate-docs")
            sys.exit(1)
        elif new_score > previous_score and not args.quiet:
            print(f"Coverage improved: {previous_score}% -> {new_score}% (+{new_score - previous_score:.1f}%)")

    if not args.quiet:
        print_summary(report)

    if args.update:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        if not args.quiet:
            print(f"Report written to {args.output}")

    if args.generate_docs:
        result = subprocess.run(
            [sys.executable, SCRIPT_DIR / "generate_google_interactions_coverage_docs.py"],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
