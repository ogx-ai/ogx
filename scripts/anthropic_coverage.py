#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Anthropic Messages API Coverage Analyzer

Uses oasdiff to compare OGX's OpenAPI spec against Anthropic's spec
and generates a coverage report showing:
- Which endpoints are implemented
- Which properties are conformant vs have issues
- A conformance score that increases as issues are fixed

The Anthropic spec is extracted from the Stainless-generated OpenAPI spec used
for SDK generation. It is not officially published at a stable URL. The current
spec URL is found in the anthropic-sdk-python repo's .stats.yml file:

    https://github.com/anthropics/anthropic-sdk-python/blob/main/.stats.yml

The openapi_spec_url field in that file points to a GCS-hosted YAML/JSON spec
that changes with every SDK release (hash-based path). To update the vendored
spec, download from that URL and run the filtering in the update instructions
below.

To update docs/static/anthropic-spec.json:

    1. Check .stats.yml for the current openapi_spec_url
    2. curl -sL <url> -o docs/static/anthropic-spec-raw.json
    3. Filter out ?beta=true paths and fix exclusiveMinimum for oasdiff compat
    4. Run: python scripts/anthropic_coverage.py --update --generate-docs

Usage:
    python scripts/anthropic_coverage.py [--update]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_spec(spec_path: Path) -> dict[str, Any]:
    """Load an OpenAPI spec from YAML or JSON."""
    content = spec_path.read_text()
    if spec_path.suffix in (".yml", ".yaml"):
        import yaml

        return yaml.safe_load(content)
    return json.loads(content)


def _count_schema_properties(schema: dict[str, Any], spec: dict[str, Any], visited: set[str] | None = None) -> int:
    """Recursively count properties in a schema, resolving $ref references."""
    if visited is None:
        visited = set()

    if not isinstance(schema, dict):
        return 0

    count = 0

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in visited:
            return 0
        visited.add(ref)

        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            resolved = spec
            for part in parts:
                resolved = resolved.get(part, {})
            count += _count_schema_properties(resolved, spec, visited)
        return count

    if "properties" in schema:
        props = schema["properties"]
        count += len(props)
        for prop_schema in props.values():
            count += _count_schema_properties(prop_schema, spec, visited)

    for key in ("allOf", "oneOf", "anyOf"):
        if key in schema:
            for sub_schema in schema[key]:
                count += _count_schema_properties(sub_schema, spec, visited)

    if "items" in schema:
        count += _count_schema_properties(schema["items"], spec, visited)

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        count += _count_schema_properties(schema["additionalProperties"], spec, visited)

    return count


def _count_endpoint_properties(spec: dict[str, Any], paths: list[str]) -> int:
    """Count total properties for the given endpoint paths in a spec."""
    total = 0

    for path in paths:
        path_item = spec.get("paths", {}).get(path, {})
        if not isinstance(path_item, dict):
            continue

        for method, operation in path_item.items():
            if method in ("parameters", "servers", "summary", "description"):
                continue
            if not isinstance(operation, dict):
                continue

            request_body = operation.get("requestBody", {})
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                json_content = content.get("application/json", {})
                if isinstance(json_content, dict) and "schema" in json_content:
                    total += _count_schema_properties(json_content["schema"], spec)

            responses = operation.get("responses", {})
            for response in responses.values():
                if isinstance(response, dict):
                    content = response.get("content", {})
                    json_content = content.get("application/json", {})
                    if isinstance(json_content, dict) and "schema" in json_content:
                        total += _count_schema_properties(json_content["schema"], spec)

            params = operation.get("parameters", [])
            total += len(params)

    return total


# Only track Anthropic Messages endpoints. Files and Models use the same path
# as OpenAI but different schemas - OGX serves OpenAI-shaped responses
# for both, so they are tracked by scripts/openai_coverage.py instead.
# Anthropic's /v1/models returns a completely different shape (capabilities,
# display_name, max_input_tokens) that OGX does not implement.
_CATEGORIES: dict[str, list[str]] = {
    "Messages": ["/v1/messages", "/v1/messages/count_tokens"],
    "Message Batches": [
        "/v1/messages/batches",
        "/v1/messages/batches/{message_batch_id}",
        "/v1/messages/batches/{message_batch_id}/cancel",
        "/v1/messages/batches/{message_batch_id}/results",
    ],
}


def _categorize_path(path: str) -> str:
    """Categorize an endpoint path using the predefined categories."""
    for category, paths in _CATEGORIES.items():
        if path in paths:
            return category
    return "Other"


def _check_oasdiff_installed() -> bool:
    """Check if oasdiff is installed and accessible."""
    return shutil.which("oasdiff") is not None


def _run_oasdiff(anthropic_spec: Path, llama_spec: Path) -> dict[str, Any]:
    """Run oasdiff and return the JSON diff."""
    if not _check_oasdiff_installed():
        print("Error: oasdiff is not installed or not in PATH")
        print()
        print("See installation instructions at:")
        print("  https://github.com/oasdiff/oasdiff#installation")
        print()
        sys.exit(1)

    result = subprocess.run(
        [
            "oasdiff",
            "diff",
            str(anthropic_spec),
            str(llama_spec),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"oasdiff failed: {result.stderr}")

    return json.loads(result.stdout) if result.stdout else {}


def _extract_issues(obj: Any, path: str = "") -> dict[str, Any]:
    """Recursively extract issues from oasdiff output."""
    result: dict[str, Any] = {
        "missing": [],
        "issues": [],
    }

    if not isinstance(obj, dict):
        return result

    if "deleted" in obj:
        deleted = obj["deleted"]
        if isinstance(deleted, list):
            for item in deleted:
                if isinstance(item, str):
                    prop_path = f"{path}.{item}" if path else item
                    prop_path = prop_path.replace(".modified.", ".")
                    result["missing"].append(prop_path)
        elif isinstance(deleted, dict):
            for location, items in deleted.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            prop_path = f"{path}.{location}.{item}" if path else f"{location}.{item}"
                            prop_path = prop_path.replace(".modified.", ".")
                            result["missing"].append(prop_path)

    if "modified" in obj and isinstance(obj["modified"], dict):
        for prop_name, prop_diff in obj["modified"].items():
            if not isinstance(prop_diff, dict):
                continue

            schema_indicators = {"enum", "type", "listOfTypes", "anyOf", "oneOf", "default", "schema"}
            if not any(key in prop_diff for key in schema_indicators):
                nested_path = f"{path}.{prop_name}" if path else prop_name
                nested = _extract_issues({"modified": prop_diff}, nested_path)
                result["missing"].extend(nested["missing"])
                result["issues"].extend(nested["issues"])
                continue

            clean_name = prop_name.replace(".modified.", ".")
            prop_path = f"{path}.{clean_name}" if path else clean_name
            prop_path = prop_path.replace(".modified.", ".")

            issue_details = []

            if "enum" in prop_diff:
                enum_diff = prop_diff["enum"]
                if enum_diff.get("enumDeleted"):
                    issue_details.append(f"Enum removed: {enum_diff.get('deleted', [])}")
                elif "deleted" in enum_diff:
                    issue_details.append(f"Enum values removed: {enum_diff['deleted']}")

            if "type" in prop_diff:
                type_diff = prop_diff["type"]
                if "deleted" in type_diff:
                    issue_details.append(f"Type removed: {type_diff['deleted']}")
                if "added" in type_diff:
                    issue_details.append(f"Type added: {type_diff['added']}")

            if "listOfTypes" in prop_diff:
                lot_diff = prop_diff["listOfTypes"]
                if "added" in lot_diff and "null" in lot_diff["added"]:
                    issue_details.append("Nullable added (Anthropic non-nullable)")
                if "deleted" in lot_diff and "null" in lot_diff["deleted"]:
                    issue_details.append("Nullable removed (Anthropic nullable)")

            if "anyOf" in prop_diff or "oneOf" in prop_diff:
                union_diff = prop_diff.get("anyOf", prop_diff.get("oneOf", {}))
                if "added" in union_diff:
                    issue_details.append(f"Union variants added: {len(union_diff['added'])}")
                if "deleted" in union_diff:
                    issue_details.append(f"Union variants removed: {len(union_diff['deleted'])}")

            if "default" in prop_diff:
                default_diff = prop_diff["default"]
                if isinstance(default_diff, dict) and "from" in default_diff:
                    issue_details.append(f"Default changed: {default_diff['from']} -> {default_diff.get('to')}")

            if issue_details:
                result["issues"].append({"property": prop_path, "details": issue_details})

            nested = _extract_issues(prop_diff, prop_path)
            result["missing"].extend(nested["missing"])
            result["issues"].extend(nested["issues"])

    for key in ["schema", "items", "properties", "content", "responses", "parameters", "requestBody"]:
        if key in obj and isinstance(obj[key], dict):
            nested_path = f"{path}.{key}" if path and key not in ("schema", "items") else path
            nested = _extract_issues(obj[key], nested_path)
            result["missing"].extend(nested["missing"])
            result["issues"].extend(nested["issues"])

    return result


def calculate_coverage(
    anthropic_spec: Path,
    llama_spec: Path,
) -> dict[str, Any]:
    """Calculate coverage metrics."""
    diff = _run_oasdiff(anthropic_spec, llama_spec)

    paths_diff = diff.get("paths", {})
    deleted_paths = paths_diff.get("deleted", [])
    modified_paths = paths_diff.get("modified", {})

    # Filter to relevant categories
    missing_endpoints = sorted([p for p in deleted_paths if _categorize_path(p) != "Other"])
    implemented_endpoints = sorted([p for p in modified_paths.keys() if _categorize_path(p) != "Other"])

    # Analyze each endpoint
    categories: dict[str, dict[str, Any]] = {}

    for path in implemented_endpoints:
        category = _categorize_path(path)
        if category not in categories:
            categories[category] = {
                "endpoints": [],
                "total_issues": 0,
                "total_missing": 0,
            }

        endpoint_diff = modified_paths[path]
        operations = endpoint_diff.get("operations", {}).get("modified", {})

        endpoint_data = {
            "path": path,
            "operations": [],
        }

        for method in sorted(operations.keys()):
            op_diff = operations[method]
            issues_data = _extract_issues(op_diff, method)

            sorted_missing = sorted(issues_data["missing"])
            sorted_issues = sorted(issues_data["issues"], key=lambda x: x["property"])

            endpoint_data["operations"].append(
                {
                    "method": method,
                    "missing_properties": sorted_missing,
                    "conformance_issues": sorted_issues,
                    "missing_count": len(sorted_missing),
                    "issues_count": len(sorted_issues),
                }
            )

            categories[category]["total_issues"] += len(sorted_issues)
            categories[category]["total_missing"] += len(sorted_missing)

        categories[category]["endpoints"].append(endpoint_data)

    # Calculate totals
    total_issues = sum(c["total_issues"] for c in categories.values())
    total_missing = sum(c["total_missing"] for c in categories.values())
    total_endpoints = len(implemented_endpoints) + len(missing_endpoints)

    # Count properties from spec
    anthropic_spec_data = _load_spec(anthropic_spec)

    total_properties = 0
    category_properties: dict[str, int] = {}

    for cat_name, cat_data in categories.items():
        cat_paths = [ep["path"] for ep in cat_data["endpoints"]]
        cat_props = _count_endpoint_properties(anthropic_spec_data, cat_paths)

        cat_problems = cat_data["total_issues"] + cat_data["total_missing"]
        category_properties[cat_name] = max(cat_props, cat_problems)
        total_properties += category_properties[cat_name]

    total_problems = total_issues + total_missing
    total_properties = max(total_properties, total_problems)

    if total_properties > 0:
        overall_score = round((1 - total_problems / total_properties) * 100, 1)
    else:
        overall_score = 100.0 if total_problems == 0 else 0.0

    report: dict[str, Any] = {
        "anthropic_spec": str(anthropic_spec),
        "llama_spec": str(llama_spec),
        "summary": {
            "endpoints": {
                "implemented": len(implemented_endpoints),
                "total": total_endpoints,
                "missing": missing_endpoints,
            },
            "conformance": {
                "score": overall_score,
                "issues": total_issues,
                "missing_properties": total_missing,
                "total_problems": total_problems,
                "total_properties": total_properties,
            },
        },
        "categories": {},
    }

    for cat_name in sorted(categories.keys()):
        cat_data = categories[cat_name]
        cat_problems = cat_data["total_issues"] + cat_data["total_missing"]
        cat_total = category_properties.get(cat_name, cat_problems)

        if cat_total > 0:
            cat_score = round((1 - cat_problems / cat_total) * 100, 1)
        else:
            cat_score = 100.0 if cat_problems == 0 else 0.0

        report["categories"][cat_name] = {
            "score": cat_score,
            "issues": cat_data["total_issues"],
            "missing_properties": cat_data["total_missing"],
            "total_properties": cat_total,
            "endpoints": sorted(cat_data["endpoints"], key=lambda x: x["path"]),
        }

    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    summary = report["summary"]

    print("\n" + "=" * 60)
    print("Anthropic Messages API Conformance Report")
    print("=" * 60)
    print()

    conf = summary["conformance"]
    print(f"Overall Conformance Score: {conf['score']}%")
    print()

    ep = summary["endpoints"]
    print(f"Endpoints: {ep['implemented']}/{ep['total']} implemented")
    if ep["missing"]:
        print(f"   Missing: {', '.join(ep['missing'][:3])}")
        if len(ep["missing"]) > 3:
            print(f"            ... and {len(ep['missing']) - 3} more")
    print()

    print("Remaining Issues")
    print("-" * 40)
    print(f"   Schema/Type issues:    {conf['issues']}")
    print(f"   Missing properties:    {conf['missing_properties']}")
    print(f"   Total to fix:          {conf['total_problems']}")
    print()

    print("Score by Category")
    print("-" * 50)
    print(f"{'Category':<20} {'Score':<10} {'Issues':<10} {'Missing':<10}")
    print("-" * 50)

    for cat_name in sorted(
        report["categories"].keys(),
        key=lambda x: report["categories"][x]["score"],
    ):
        cat = report["categories"][cat_name]
        print(f"{cat_name:<20} {cat['score']:>5}%    {cat['issues']:<10} {cat['missing_properties']:<10}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Anthropic Messages API conformance analyzer")
    parser.add_argument(
        "--anthropic-spec",
        type=Path,
        default=Path("docs/static/anthropic-spec.json"),
        help="Path to Anthropic spec",
    )
    parser.add_argument(
        "--llama-spec",
        type=Path,
        default=Path("docs/static/ogx-spec.yaml"),
        help="Path to OGX spec",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/static/anthropic-coverage.json"),
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

    previous_score: float | None = None
    if args.check_regression and args.output.exists():
        try:
            with open(args.output) as f:
                previous_report = json.load(f)
                previous_score = previous_report.get("summary", {}).get("conformance", {}).get("score")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error: could not load previous report: {e}")
            sys.exit(1)

    try:
        report = calculate_coverage(args.anthropic_spec, args.llama_spec)
    except FileNotFoundError:
        print("Error: oasdiff not found. Install with: go install github.com/tufin/oasdiff@latest")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    new_score = report["summary"]["conformance"]["score"]

    if args.check_regression and previous_score is not None:
        if new_score < previous_score:
            print(f"Coverage regression detected: {previous_score}% -> {new_score}%")
            print(f"Coverage decreased by {previous_score - new_score:.1f} percentage points")
            print()
            print("To fix this, ensure your changes don't reduce Anthropic API conformance.")
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
        script_dir = Path(__file__).parent
        result = subprocess.run(
            [sys.executable, script_dir / "generate_anthropic_coverage_docs.py"],
            cwd=script_dir.parent,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
