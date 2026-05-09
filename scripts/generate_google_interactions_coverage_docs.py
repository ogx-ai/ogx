#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Generates Google Interactions API coverage documentation from the coverage JSON file.

Usage:
    python scripts/generate_google_interactions_coverage_docs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_COVERAGE_JSON = REPO_ROOT / "docs" / "static" / "google-interactions-coverage.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "docs" / "api-google-interactions" / "conformance.mdx"


def generate_docs(coverage_path: Path, output_path: Path) -> None:
    """Generate markdown documentation from coverage JSON."""
    with open(coverage_path) as f:
        coverage = json.load(f)

    summary = coverage["summary"]
    sections = coverage["sections"]

    # Sort sections by score (lowest first)
    sorted_sections = sorted(sections, key=lambda s: s["score"])

    lines = [
        "---",
        "title: Google Interactions API Conformance",
        "description: Coverage status of OGX against the Google Interactions API specification",
        "sidebar_label: Interactions Conformance",
        "sidebar_position: 5",
        "---",
        "",
        "# Google Interactions API Coverage Report",
        "",
        f"Spec version: `{coverage.get('spec_version', 'unknown')}`",
        "",
        "This page provides a detailed breakdown of OGX's coverage of the",
        "[Google Interactions API](https://ai.google.dev/gemini-api/docs/interactions) specification.",
        "The coverage score increases as missing features are implemented.",
        "",
        ":::info Auto-generated",
        "This documentation is auto-generated from the Google Interactions API specification comparison.",
        ":::",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Overall Coverage Score** | {summary['overall_score']}% |",
        f"| **Total Items Implemented** | {summary['total_implemented']}/{summary['total_google_items']} |",
        f"| **Total Missing** | {summary['total_missing']} |",
        "",
        "## Section Scores",
        "",
        "Sections are sorted by coverage score (lowest first, needing most attention).",
        "",
        "| Section | Score | Implemented | Total | Missing |",
        "|---------|-------|-------------|-------|---------|",
    ]

    for section in sorted_sections:
        lines.append(
            f"| {section['section']} | {section['score']}% "
            f"| {section['implemented']} | {section['google_total']} "
            f"| {section['missing_count']} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Breakdown",
            "",
            "Below is a detailed breakdown of what is implemented and what is missing for each section.",
            "",
        ]
    )

    for section in sorted_sections:
        lines.append(f"### {section['section']}")
        lines.append("")
        lines.append(
            f"**Score:** {section['score']}% · **Implemented:** {section['implemented']}/{section['google_total']}"
        )
        lines.append("")

        if section["supported"]:
            lines.append("<details>")
            lines.append(f"<summary>Implemented ({section['implemented']})</summary>")
            lines.append("")
            for item in section["supported"]:
                lines.append(f"- `{item}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if section["missing"]:
            lines.append("<details>")
            lines.append(f"<summary>Missing ({section['missing_count']})</summary>")
            lines.append("")
            for item in section["missing"]:
                lines.append(f"- `{item}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if section.get("extra_in_ogx"):
            lines.append("<details>")
            lines.append(f"<summary>Extra in OGX ({len(section['extra_in_ogx'])})</summary>")
            lines.append("")
            lines.append("These properties are in the OGX implementation but not in the Google spec:")
            lines.append("")
            for item in section["extra_in_ogx"]:
                lines.append(f"- `{item}`")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.extend(
        [
            "## How to Improve Coverage",
            "",
            "To improve coverage scores:",
            "",
            "1. **Add Missing Properties**: Implement missing fields in request/response models"
            " in `src/ogx_api/interactions/models.py`",
            "2. **Add Content Types**: Support additional content types beyond text"
            " (images, audio, function calls, etc.)",
            "3. **Add Tool Support**: Implement tool declarations (Function, GoogleSearch, CodeExecution, etc.)",
            "4. **Add Missing Endpoints**: Implement GET, DELETE, and Cancel endpoints",
            "",
            "Run the coverage analyzer to check your progress:",
            "",
            "```bash",
            "python scripts/google_interactions_coverage.py --update --generate-docs",
            "```",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Generated documentation: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Google Interactions coverage documentation")
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
        print("Run 'python scripts/google_interactions_coverage.py --update' first")
        return 1

    generate_docs(args.coverage_json, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
