#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Generates Anthropic Messages API coverage documentation from the coverage JSON file.

Usage:
    python scripts/generate_anthropic_coverage_docs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_COVERAGE_JSON = REPO_ROOT / "docs" / "static" / "anthropic-coverage.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "docs" / "api-anthropic-messages" / "conformance.mdx"


def generate_docs(
    coverage_path: Path,
    output_path: Path,
) -> None:
    """Generate markdown documentation from coverage JSON."""
    with open(coverage_path) as f:
        coverage = json.load(f)

    summary = coverage["summary"]
    categories = coverage["categories"]

    sorted_categories = sorted(categories.items(), key=lambda x: x[1]["score"])

    lines = [
        "---",
        "title: Anthropic Messages API Conformance",
        "description: Detailed conformance status of OGX against the Anthropic Messages API specification",
        "sidebar_label: Conformance",
        "sidebar_position: 2",
        "---",
        "",
        "# Anthropic Messages API Conformance Report",
        "",
        "This page provides a detailed breakdown of OGX's conformance to the Anthropic Messages API specification.",
        "The conformance score increases as schema issues are fixed and missing properties are implemented.",
        "",
        ":::info Auto-generated",
        "This documentation is auto-generated from the Anthropic Messages API specification comparison.",
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
        "",
        "## Category Scores",
        "",
        "Categories are sorted by conformance score (lowest first, needing most attention).",
        "",
        "| Category | Score | Properties | Issues | Missing |",
        "|----------|-------|------------|--------|---------|",
    ]

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
            "The following Anthropic API endpoints are not yet implemented in OGX:",
            "",
        ]
    )

    missing = summary["endpoints"]["missing"]
    if missing:
        grouped: dict[str, list[str]] = {}
        for endpoint in missing:
            parts = endpoint.strip("/").split("/")
            prefix = parts[1] if len(parts) > 1 else parts[0]
            grouped.setdefault(prefix, []).append(endpoint)

        for prefix in sorted(grouped.keys()):
            lines.append(f"### /{prefix}")
            lines.append("")
            for endpoint in sorted(grouped[prefix]):
                lines.append(f"- `{endpoint}`")
            lines.append("")
    else:
        lines.append("All Anthropic API endpoints are implemented!")
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
                        clean_prop = prop.replace(f"{op['method']}.", "")
                        lines.append(f"- `{clean_prop}`")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

                if op["conformance_issues"]:
                    lines.append("<details>")
                    lines.append(f"<summary>Schema Issues ({op['issues_count']})</summary>")
                    lines.append("")
                    lines.append("| Property | Issues |")
                    lines.append("|----------|--------|")
                    for issue in op["conformance_issues"]:
                        clean_prop = issue["property"].replace(f"{op['method']}.", "")
                        details = "; ".join(issue["details"])
                        details = details.replace("|", "\\|")
                        details = details.replace("<", "&lt;").replace(">", "&gt;")
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
            "1. **Fix Schema Issues**: Update Pydantic models in `src/ogx_api/messages/` to match Anthropic's schema",
            "2. **Add Missing Properties**: Implement missing fields in request/response models",
            "3. **Add Missing Endpoints**: Implement endpoints listed in the Missing Endpoints section",
            "",
            "Run the coverage analyzer to check your progress:",
            "",
            "```bash",
            "python scripts/anthropic_coverage.py --update",
            "```",
            "",
            "Then regenerate this documentation:",
            "",
            "```bash",
            "python scripts/generate_anthropic_coverage_docs.py",
            "```",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Generated documentation: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Anthropic coverage documentation")
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
        print(f"Coverage JSON not found: {args.coverage_json}")
        print("Run 'python scripts/anthropic_coverage.py --update' first")
        return 1

    generate_docs(args.coverage_json, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
