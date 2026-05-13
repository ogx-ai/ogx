#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Update constraint-dependencies in pyproject.toml for Dependabot PRs.

Dependabot's uv ecosystem only modifies uv.lock directly. This script
updates the >= lower bound in [tool.uv] constraint-dependencies so that
pyproject.toml remains the source of truth.
"""

import argparse
import re
import sys
from pathlib import Path


def parse_version(version_str: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version_str.split("."))


def normalize_pkg_pattern(pkg_name: str) -> str:
    """Convert a package name into a regex pattern matching any PEP 503 equivalent."""
    return re.sub(r"[-_.]", "[-_.]", pkg_name.lower())


def find_constraint_section(lines: list[str]) -> tuple[int, int] | None:
    """Find the start and end line indices of the constraint-dependencies array."""
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^constraint-dependencies\s*=\s*\[", line):
            start = i
            continue
        if start is not None and line.rstrip().rstrip(",").endswith("]"):
            return start, i
    return None


def find_constraint_line(lines: list[str], pkg_name: str) -> int | None:
    section = find_constraint_section(lines)
    if section is None:
        return None
    start, end = section
    pattern = re.compile(rf'^\s*"{normalize_pkg_pattern(pkg_name)}', re.IGNORECASE)
    for i in range(start, end + 1):
        if pattern.match(lines[i]):
            return i
    return None


def update_constraint(line: str, pkg_name: str, new_version: str) -> tuple[str, bool, str]:
    """Update the >= lower bound in a constraint-dependencies line.

    Returns (new_line, changed, reason).
    """
    pkg_pattern = normalize_pkg_pattern(pkg_name)
    lower_bound_pattern = re.compile(rf'("{pkg_pattern}>=)([\d]+(?:\.[\d]+)*)', re.IGNORECASE)

    match = lower_bound_pattern.search(line)
    if not match:
        return line, False, f"no >= lower bound for {pkg_name}"

    old_version = match.group(2)
    if parse_version(new_version) <= parse_version(old_version):
        return line, False, (f"{pkg_name}: new version {new_version} <= current floor {old_version}")

    upper_bound_match = re.search(rf'"{pkg_pattern}>=[^"]*,<([\d]+(?:\.[\d]+)*)"', line, re.IGNORECASE)
    if upper_bound_match:
        upper_version = upper_bound_match.group(1)
        if parse_version(new_version) >= parse_version(upper_version):
            return line, False, (f"{pkg_name}: new version {new_version} >= upper bound <{upper_version}, skipping")

    new_line = lower_bound_pattern.sub(lambda m: m.group(1) + new_version, line)
    return new_line, True, (f"{pkg_name}: updated >= floor from {old_version} to {new_version}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update constraint-dependencies in pyproject.toml")
    parser.add_argument("--dependency-name", required=True)
    parser.add_argument("--dependency-version", required=True)
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: pyproject.toml)",
    )
    args = parser.parse_args()

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        return 1

    content = pyproject_path.read_text()
    lines = content.splitlines(keepends=True)

    line_idx = find_constraint_line(lines, args.dependency_name)
    if line_idx is None:
        print(f"SKIP: {args.dependency_name} not found in constraint-dependencies")
        print("updated=false")
        return 0

    new_line, changed, reason = update_constraint(lines[line_idx], args.dependency_name, args.dependency_version)

    if not changed:
        print(f"SKIP: {reason}")
        print("updated=false")
        return 0

    lines[line_idx] = new_line
    pyproject_path.write_text("".join(lines))
    print(f"UPDATED: {reason}")
    print("updated=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
