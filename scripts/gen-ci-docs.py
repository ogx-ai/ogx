#!/usr/bin/env python
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent


def parse_workflow_file(file_path):
    """Parse a workflow YAML file and extract name and run-name."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        name = content.get("name", "Unknown")
        # run-name is optional in GitHub Actions workflows
        run_name = content.get("run-name", name)

        return name, run_name
    except Exception as e:
        raise Exception(f"Error parsing {file_path}") from e


def generate_ci_docs():
    """Generate the CI documentation README.md file."""

    # Define paths
    workflows_dir = REPO_ROOT / ".github/workflows"
    readme_path = workflows_dir / "README.md"

    # Header section to preserve
    header = """# OGX CI

OGX uses GitHub Actions for Continuous Integration (CI). Below is a table detailing what CI the project includes and the purpose.

| Name | File | Purpose |
| ---- | ---- | ------- |
"""

    # Get all .yml files in workflows directory
    yml_files = []
    for file_path in workflows_dir.glob("*.yml"):
        yml_files.append(file_path)

    # Sort files alphabetically for consistent output
    yml_files.sort(key=lambda x: x.name)

    # Generate table rows
    table_rows = []
    for file_path in yml_files:
        name, run_name = parse_workflow_file(file_path)
        filename = file_path.name

        # Create markdown link in the format [filename.yml](filename.yml)
        file_link = f"[{filename}]({filename})"

        # Create table row
        row = f"| {name} | {file_link} | {run_name} |"
        table_rows.append(row)

    # Combine header and table rows
    content = header + "\n".join(table_rows) + "\n"

    # Write to README.md
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated {readme_path} with {len(table_rows)} workflow entries")


if __name__ == "__main__":
    generate_ci_docs()
