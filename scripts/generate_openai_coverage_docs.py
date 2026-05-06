#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Generates OpenAI API coverage documentation from the coverage JSON file.

Usage:
    python scripts/generate_openai_coverage_docs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_COVERAGE_JSON = REPO_ROOT / "docs" / "static" / "openai-coverage.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "docs" / "api-openai" / "conformance.mdx"

# Categories whose properties can be cross-referenced with integration tests
_TESTABLE_CATEGORIES = {"Responses", "Conversations"}


def _extract_property_name(property_path: str) -> str | None:
    """Extract the leaf property name from a conformance property path.

    E.g. 'POST.requestBody.content.application/json.properties.temperature' -> 'temperature'
         'POST.responses.200.content.application/json.properties.output.items' -> 'output'
    """
    marker = ".properties."
    idx = property_path.rfind(marker)
    if idx == -1:
        return None
    remainder = property_path[idx + len(marker) :]
    # Take the first segment (before any sub-path like '.items')
    return remainder.split(".")[0]


def _load_test_coverage() -> tuple[set[str], float | None, dict[str, tuple[int, int]]]:
    """Load test coverage data: tested properties, overall score, and per-category counts."""
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from responses_test_coverage import get_tested_property_names, run_coverage

        features = run_coverage()
        tested = get_tested_property_names(features)
        total = len(features)
        covered = sum(1 for f in features if f.covered)
        score = (covered / total * 100) if total > 0 else 0.0

        cat_counts: dict[str, tuple[int, int]] = {}
        for f in features:
            c, t = cat_counts.get(f.category, (0, 0))
            cat_counts[f.category] = (c + (1 if f.covered else 0), t + 1)

        return tested, round(score, 1), cat_counts
    except ImportError:
        return set(), None, {}
    finally:
        if str(SCRIPT_DIR) in sys.path:
            sys.path.remove(str(SCRIPT_DIR))


def generate_docs(
    coverage_path: Path,
    output_path: Path,
    tested_properties: set[str] | None = None,
    test_score: float | None = None,
    test_cat_counts: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Generate markdown documentation from coverage JSON."""
    with open(coverage_path) as f:
        coverage = json.load(f)

    if tested_properties is None:
        tested_properties = set()

    summary = coverage["summary"]
    categories = coverage["categories"]

    # Sort categories by score (lowest first)
    sorted_categories = sorted(categories.items(), key=lambda x: x[1]["score"])

    lines = [
        "---",
        "title: OpenAI API Conformance",
        "description: Detailed conformance status of OGX against the OpenAI API specification",
        "sidebar_label: Conformance",
        "sidebar_position: 2",
        "---",
        "",
        "# OpenAI API Conformance Report",
        "",
        "This page provides a detailed breakdown of OGX's conformance to the OpenAI API specification.",
        "The conformance score increases as schema issues are fixed and missing properties are implemented.",
        "",
        ":::info Auto-generated",
        "This documentation is auto-generated from the OpenAI API specification comparison.",
        ":::",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Overall Conformance Score** | {summary['conformance']['score']}% |",
        f"| **Endpoints Implemented** | {summary['endpoints']['implemented']}/{summary['endpoints']['total']} |",
        f"| **Total Properties Checked** | {summary['conformance'].get('total_properties', 'N/A')} |",
        f"| **Schema/Type Issues** | {summary['conformance']['issues']} |",
        f"| **Missing Properties** | {summary['conformance']['missing_properties']} |",
        f"| **Total Issues to Fix** | {summary['conformance']['total_problems']} |",
    ]

    if test_score is not None:
        lines.extend(
            [
                "",
                "## Integration Test Coverage",
                "",
                f"Overall Test Coverage Score: {test_score}%",
                "",
                "| Category | Covered | Total | Score |",
                "|----------|---------|-------|-------|",
            ]
        )
        if test_cat_counts:
            for cat_name, (covered, total) in sorted(test_cat_counts.items(), key=lambda x: x[0]):
                cat_score = (covered / total * 100) if total > 0 else 0
                lines.append(f"| {cat_name} | {covered} | {total} | {cat_score:.1f}% |")

    lines.extend(
        [
            "",
            "## Category Scores",
            "",
            "Categories are sorted by conformance score (lowest first, needing most attention).",
            "",
            "| Category | Score | Properties | Issues | Missing |",
            "|----------|-------|------------|--------|---------|",
        ]
    )

    for cat_name, cat_data in sorted_categories:
        total_props = cat_data.get("total_properties", "N/A")
        lines.append(
            f"| {cat_name} | {cat_data['score']}% | {total_props} | {cat_data['issues']} | {cat_data['missing_properties']} |"
        )

    lines.extend(
        [
            "",
            "## Missing Endpoints",
            "",
            "The following OpenAI API endpoints are not yet implemented in OGX:",
            "",
        ]
    )

    missing = summary["endpoints"]["missing"]
    if missing:
        # Group by prefix for better readability
        grouped: dict[str, list[str]] = {}
        for endpoint in missing:
            prefix = endpoint.split("/")[1] if "/" in endpoint else "other"
            grouped.setdefault(prefix, []).append(endpoint)

        for prefix in sorted(grouped.keys()):
            lines.append(f"### /{prefix}")
            lines.append("")
            for endpoint in sorted(grouped[prefix]):
                lines.append(f"- `{endpoint}`")
            lines.append("")
    else:
        lines.append("All OpenAI API endpoints are implemented! 🎉")
        lines.append("")

    lines.extend(
        [
            "## Detailed Category Breakdown",
            "",
            "Below is a detailed breakdown of conformance issues and missing properties for each category.",
            "",
        ]
    )

    for cat_name, cat_data in sorted_categories:
        if cat_data["issues"] == 0 and cat_data["missing_properties"] == 0:
            continue

        lines.append(f"### {cat_name}")
        lines.append("")
        lines.append(
            f"**Score:** {cat_data['score']}% · "
            f"**Issues:** {cat_data['issues']} · "
            f"**Missing:** {cat_data['missing_properties']}"
        )
        lines.append("")

        for endpoint in cat_data["endpoints"]:
            endpoint_has_issues = False
            for op in endpoint["operations"]:
                if op["conformance_issues"] or op["missing_properties"]:
                    endpoint_has_issues = True
                    break

            if not endpoint_has_issues:
                continue

            lines.append(f"#### `{endpoint['path']}`")
            lines.append("")

            for op in endpoint["operations"]:
                if not op["conformance_issues"] and not op["missing_properties"]:
                    continue

                lines.append(f"**{op['method']}**")
                lines.append("")

                if op["missing_properties"]:
                    lines.append("<details>")
                    lines.append(f"<summary>Missing Properties ({op['missing_count']})</summary>")
                    lines.append("")
                    for prop in op["missing_properties"]:
                        # Clean up the property path for readability
                        clean_prop = prop.replace(f"{op['method']}.", "")
                        lines.append(f"- `{clean_prop}`")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

                if op["conformance_issues"]:
                    show_tested = cat_name in _TESTABLE_CATEGORIES and tested_properties
                    lines.append("<details>")
                    lines.append(f"<summary>Schema Issues ({op['issues_count']})</summary>")
                    lines.append("")
                    if show_tested:
                        lines.append("| Property | Issues | Tested |")
                        lines.append("|----------|--------|--------|")
                    else:
                        lines.append("| Property | Issues |")
                        lines.append("|----------|--------|")
                    for issue in op["conformance_issues"]:
                        clean_prop = issue["property"].replace(f"{op['method']}.", "")
                        details = "; ".join(issue["details"])
                        # Escape pipe characters in details
                        details = details.replace("|", "\\|")
                        # Escape < and > for MDX compatibility (prevents JSX parse errors)
                        details = details.replace("<", "&lt;").replace(">", "&gt;")
                        # Escape curly braces for MDX (prevents acorn parse errors)
                        details = details.replace("{", "\\{").replace("}", "\\}")
                        if show_tested:
                            prop_name = _extract_property_name(issue["property"])
                            is_tested = prop_name in tested_properties if prop_name else False
                            tested_indicator = "Yes" if is_tested else "No"
                            lines.append(f"| `{clean_prop}` | {details} | {tested_indicator} |")
                        else:
                            lines.append(f"| `{clean_prop}` | {details} |")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

    lines.extend(
        [
            "## How to Improve Conformance",
            "",
            "To improve conformance scores:",
            "",
            "1. **Fix Schema Issues**: Update Pydantic models in `src/ogx_api/` to match OpenAI's schema",
            "2. **Add Missing Properties**: Implement missing fields in response models",
            "3. **Add Missing Endpoints**: Implement endpoints listed in the Missing Endpoints section",
            "",
            "Run the coverage analyzer to check your progress:",
            "",
            "```bash",
            "python scripts/openai_coverage.py --update",
            "```",
            "",
            "Then regenerate this documentation:",
            "",
            "```bash",
            "python scripts/generate_openai_coverage_docs.py",
            "```",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"✅ Generated documentation: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI coverage documentation")
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help="Path to coverage JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output markdown file path",
    )
    args = parser.parse_args()

    if not args.coverage_json.exists():
        print(f"❌ Coverage JSON not found: {args.coverage_json}")
        print("Run 'python scripts/openai_coverage.py --update' first")
        return 1

    tested_properties, test_score, test_cat_counts = _load_test_coverage()
    if tested_properties:
        print(f"Found {len(tested_properties)} tested properties from integration tests")
    if test_score is not None:
        print(f"Integration test coverage score: {test_score}%")

    generate_docs(
        args.coverage_json,
        args.output,
        tested_properties=tested_properties,
        test_score=test_score,
        test_cat_counts=test_cat_counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
