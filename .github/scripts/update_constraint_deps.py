#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Update dependency version floors in pyproject.toml for Dependabot PRs.

Dependabot's uv ecosystem only modifies uv.lock directly. This script
keeps pyproject.toml in sync by updating the >= lower bound wherever the
dependency is declared:

1. If found in a regular dependency array ([project] dependencies,
   [project.optional-dependencies], [dependency-groups]) — update in place.
2. Else if found in [tool.uv] constraint-dependencies — update in place.
3. Else — add a new entry to constraint-dependencies.
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


def find_dependency_lines(lines: list[str], pkg_name: str) -> list[int]:
    """Find all dependency lines for a package outside constraint-dependencies."""
    constraint_section = find_constraint_section(lines)
    exclude_start, exclude_end = constraint_section if constraint_section else (-1, -1)

    pattern = re.compile(rf'^\s*"{normalize_pkg_pattern(pkg_name)}[\[>=<,"\s]', re.IGNORECASE)
    matches = []
    for i, line in enumerate(lines):
        if exclude_start <= i <= exclude_end:
            continue
        if pattern.match(line):
            matches.append(i)
    return matches


def _canonical_sort_key(line: str) -> str:
    """Extract the package name from a constraint line for alphabetical sorting."""
    m = re.match(r'^\s*"([^>=<\[" ]+)', line)
    return m.group(1).lower().replace("-", "").replace("_", "").replace(".", "") if m else ""


def insert_constraint(lines: list[str], pkg_name: str, version: str) -> tuple[list[str], bool, str]:
    """Insert a new constraint-dependencies entry in alphabetical order.

    Returns (new_lines, changed, reason).
    """
    section = find_constraint_section(lines)
    if section is None:
        return lines, False, "constraint-dependencies section not found in pyproject.toml"

    start, end = section

    new_entry = f'    "{pkg_name}>={version}",\n'
    new_key = _canonical_sort_key(new_entry)

    insert_at = end
    for i in range(start + 1, end + 1):
        if _canonical_sort_key(lines[i]) > new_key:
            insert_at = i
            break

    lines.insert(insert_at, new_entry)
    return lines, True, f"{pkg_name}: added to constraint-dependencies with >={version}"


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

    constraint_idx = find_constraint_line(lines, args.dependency_name)
    dep_indices = find_dependency_lines(lines, args.dependency_name)

    # 1. If in constraint-dependencies, that's the authoritative version spec — update there
    if constraint_idx is not None:
        new_line, changed, reason = update_constraint(
            lines[constraint_idx], args.dependency_name, args.dependency_version
        )
        if not changed:
            print(f"SKIP: {reason}")
            print("updated=false")
            return 0
        lines[constraint_idx] = new_line
        pyproject_path.write_text("".join(lines))
        print(f"UPDATED: {reason}")
        print("updated=true")
        return 0

    # 2. Try updating >= floors in regular dependency arrays.
    #    If the dep is declared here (even without a >= floor), stop — don't
    #    add a redundant entry to constraint-dependencies.
    if dep_indices:
        any_changed = False
        skip_reasons = []
        for idx in dep_indices:
            new_line, changed, reason = update_constraint(lines[idx], args.dependency_name, args.dependency_version)
            if changed:
                lines[idx] = new_line
                any_changed = True
                print(f"UPDATED (dependencies): {reason}")
            else:
                skip_reasons.append(reason)
        if any_changed:
            pyproject_path.write_text("".join(lines))
            print("updated=true")
        else:
            for r in skip_reasons:
                print(f"SKIP (dependencies): {r}")
            print("updated=false")
        return 0

    # 3. Not found anywhere — add to constraint-dependencies
    lines, changed, reason = insert_constraint(lines, args.dependency_name, args.dependency_version)
    if not changed:
        print(f"SKIP: {reason}")
        print("updated=false")
        return 0
    pyproject_path.write_text("".join(lines))
    print(f"ADDED: {reason}")
    print("updated=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
