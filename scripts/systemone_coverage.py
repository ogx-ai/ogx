#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
TypeSafe SystemOne API Conformance Analyzer

Uses oasdiff to compare OGX's OpenAPI spec (POST /v1/systemone) against
TypeSafe's published spec and generates a conformance report showing how
closely our SystemOne judgment endpoint matches TypeSafe's wire format. The
score rises as drift is fixed and `--check-regression` fails if it drops below
the committed baseline.

The reference spec is vendored from https://api.typesafe.ai/openapi.json
(TypeSafe API version 0.2.0). To refresh it:

    curl -sL https://api.typesafe.ai/openapi.json -o docs/static/typesafe-spec.json
    python scripts/systemone_coverage.py --update

Scoping: only POST /v1/systemone is checked. GET /v1/models is intentionally
excluded - OGX serves its standard multi-provider model list (OpenAI/Anthropic/
Google shapes) there, not TypeSafe's ModelMetadataList, so it is tracked by the
OpenAI/Anthropic coverage checks instead.

Deliberate divergences from TypeSafe that are normalized away before comparison
so they do not count against conformance:
- Our ``SystemOne*`` schema-name prefix (vs TypeSafe's bare names) is renamed.
- Cosmetic fields (description/summary/title/examples) have no wire impact.
- Safety bounds we add on criteria/scores (maxLength/minimum/maximum).
- ``additionalProperties: false`` (we close the API) and our optional
  ``type``/``usage`` defaults (we are a lenient superset, still wire-compatible).

Known accepted gap (not normalized away): TypeSafe declares a 422 validation-error
response on /v1/systemone, while OGX returns 400 for validation errors - a
convention used consistently across the whole API. This is the sole reason the
baseline is below 100%. Adopting 422 for validation errors is a separate,
API-wide change.

Remaining drift - a missing/renamed/retyped field, a changed type, or an enum
change - lowers the score and is caught by the regression check.

Usage:
    python scripts/systemone_coverage.py [--update]
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

# Our SystemOne schemas carry a SystemOne* prefix; TypeSafe's reference uses
# the bare names. Rename ours to theirs before comparing so the diff reflects
# wire structure, not naming. None of the target names collide with an existing
# non-SystemOne schema in our spec. SystemOneRequest/SystemOneResponse already
# match TypeSafe and are left unchanged.
_RENAME: dict[str, str] = {
    "SystemOneNoulQuestion": "NoulQuestion",
    "SystemOneChoiceQuestion": "ChoiceQuestion",
    "SystemOneScoreQuestion": "ScoreQuestion",
    "SystemOneNoulAnswer": "NoulAnswer",
    "SystemOneChoiceAnswer": "ChoiceAnswer",
    "SystemOneScoreAnswer": "ScoreAnswer",
    "SystemOneNoulCriteria": "NoulCriteria",
    "SystemOneUsage": "Usage",
}

# Schema fields that do not affect wire conformance for a clone implementation:
# cosmetic text, our added safety bounds, and the structural choices where we are
# a deliberately lenient superset (closed API, defaulted type/usage).
_NOISE_KEYS: tuple[str, ...] = (
    "description",
    "summary",
    "title",
    "examples",
    "example",
    "externalDocs",
    "deprecated",
    "default",
    "maxLength",
    "minLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "additionalProperties",
    "required",
    "discriminator",
)

# Only the SystemOne judgment endpoint is conformed to TypeSafe (see module doc).
_CATEGORIES: dict[str, list[str]] = {
    "SystemOne": ["/v1/systemone"],
}


def _load_spec(spec_path: Path) -> dict[str, Any]:
    """Load an OpenAPI spec from YAML or JSON."""
    content = spec_path.read_text()
    if spec_path.suffix in (".yml", ".yaml"):
        return yaml.safe_load(content)
    return json.loads(content)


def _rename_systemone_schemas(spec: dict[str, Any]) -> None:
    """Rename our SystemOne* schema keys to TypeSafe's bare names in place."""
    schemas = spec.get("components", {}).get("schemas", {})
    if not isinstance(schemas, dict):
        return
    spec["components"]["schemas"] = {_RENAME.get(name, name): schema for name, schema in schemas.items()}


def _rename_refs(node: Any) -> None:
    """Rewrite $ref strings to the renamed schema names, in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str) and value.startswith("#/components/schemas/"):
                name = value.rsplit("/", 1)[-1]
                node[key] = f"#/components/schemas/{_RENAME.get(name, name)}"
            else:
                _rename_refs(value)
    elif isinstance(node, list):
        for item in node:
            _rename_refs(item)


def _strip_noise(node: Any) -> None:
    """Remove non-conformance fields (cosmetic text, caps, lenient-superset keys)."""
    if isinstance(node, dict):
        for key in _NOISE_KEYS:
            node.pop(key, None)
        for value in node.values():
            _strip_noise(value)
    elif isinstance(node, list):
        for item in node:
            _strip_noise(item)


def _normalize_spec(spec: dict[str, Any], rename_systemone: bool) -> dict[str, Any]:
    """Return a copy of a spec normalized for conformance comparison."""
    normalized = copy.deepcopy(spec)
    if rename_systemone:
        _rename_systemone_schemas(normalized)
        _rename_refs(normalized)
    _strip_noise(normalized)
    return normalized


def _write_temp_spec(spec: dict[str, Any], suffix: str) -> Path:
    """Write a normalized spec to a temp file for oasdiff to read."""
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as temp_file:
        if suffix in (".yml", ".yaml"):
            yaml.safe_dump(spec, temp_file, sort_keys=False)
        else:
            json.dump(spec, temp_file, indent=2)
            temp_file.write("\n")
        return Path(temp_file.name)


def _categorize_path(path: str) -> str:
    """Categorize an endpoint path using the predefined categories."""
    for category, paths in _CATEGORIES.items():
        if path in paths:
            return category
    return "Other"


def _check_oasdiff_installed() -> bool:
    """Check if oasdiff is installed and accessible."""
    return shutil.which("oasdiff") is not None


def _run_oasdiff(typesafe_spec: Path, llama_spec: Path) -> dict[str, Any]:
    """Run oasdiff and return the JSON diff."""
    if not _check_oasdiff_installed():
        print("Error: oasdiff is not installed or not in PATH")
        print()
        print("See installation instructions at:")
        print("  https://github.com/oasdiff/oasdiff#installation")
        print()
        sys.exit(1)

    result = subprocess.run(
        ["oasdiff", "diff", str(typesafe_spec), str(llama_spec), "--format", "json"],
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
                    issue_details.append("Nullable added (TypeSafe non-nullable)")
                if "deleted" in lot_diff and "null" in lot_diff["deleted"]:
                    issue_details.append("Nullable removed (TypeSafe nullable)")

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


def calculate_coverage(typesafe_spec: Path, llama_spec: Path) -> dict[str, Any]:
    """Calculate SystemOne conformance metrics against the TypeSafe reference."""
    ts_normalized = _normalize_spec(_load_spec(typesafe_spec), rename_systemone=False)
    ogx_normalized = _normalize_spec(_load_spec(llama_spec), rename_systemone=True)
    ts_tmp = _write_temp_spec(ts_normalized, ".json")
    ogx_tmp = _write_temp_spec(ogx_normalized, ".json")

    try:
        diff = _run_oasdiff(ts_tmp, ogx_tmp)

        paths_diff = diff.get("paths", {})
        deleted_paths = paths_diff.get("deleted", [])
        modified_paths = paths_diff.get("modified", {})

        missing_endpoints = sorted([p for p in deleted_paths if _categorize_path(p) != "Other"])
        implemented_endpoints = sorted([p for p in modified_paths.keys() if _categorize_path(p) != "Other"])

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

        total_issues = sum(c["total_issues"] for c in categories.values())
        total_missing = sum(c["total_missing"] for c in categories.values())
        total_endpoints = len(implemented_endpoints) + len(missing_endpoints)

        total_properties = 0
        category_properties: dict[str, int] = {}

        for cat_name, cat_data in categories.items():
            cat_paths = [ep["path"] for ep in cat_data["endpoints"]]
            cat_props = _count_endpoint_properties(ts_normalized, cat_paths)

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
            "typesafe_spec": str(typesafe_spec),
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
    finally:
        ts_tmp.unlink(missing_ok=True)
        ogx_tmp.unlink(missing_ok=True)


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    summary = report["summary"]

    print("\n" + "=" * 60)
    print("TypeSafe SystemOne API Conformance Report")
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

    for cat_name in sorted(report["categories"].keys()):
        for endpoint in report["categories"][cat_name]["endpoints"]:
            for op in endpoint["operations"]:
                for issue in op["conformance_issues"]:
                    print(f"   {endpoint['path']} {op['method']} {issue['property']}: {', '.join(issue['details'])}")
                for missing in op["missing_properties"]:
                    print(f"   {endpoint['path']} {op['method']} missing: {missing}")

    print()


def main():
    parser = argparse.ArgumentParser(description="TypeSafe SystemOne API conformance analyzer")
    parser.add_argument(
        "--typesafe-spec",
        type=Path,
        default=Path("docs/static/typesafe-spec.json"),
        help="Path to TypeSafe reference spec",
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
        default=Path("docs/static/systemone-coverage.json"),
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
        "--check-regression",
        action="store_true",
        help="Fail if conformance score decreases compared to existing report",
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
        report = calculate_coverage(args.typesafe_spec, args.llama_spec)
    except FileNotFoundError:
        print("Error: oasdiff not found. Install with: go install github.com/tufin/oasdiff@latest")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    new_score = report["summary"]["conformance"]["score"]

    if args.check_regression and previous_score is not None:
        if new_score < previous_score:
            print(f"Conformance regression detected: {previous_score}% -> {new_score}%")
            print(f"Conformance decreased by {previous_score - new_score:.1f} percentage points")
            print()
            print("To fix this, ensure your changes don't reduce SystemOne conformance with TypeSafe.")
            print("If this is intentional, update the coverage baseline with:")
            print(f"  python {__file__} --update")
            sys.exit(1)
        elif new_score > previous_score and not args.quiet:
            print(f"Conformance improved: {previous_score}% -> {new_score}% (+{new_score - previous_score:.1f}%)")

    if not args.quiet:
        print_summary(report)

    if args.update:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        if not args.quiet:
            print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
