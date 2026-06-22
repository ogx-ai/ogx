#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Update pip_packages version floors in provider registry files for Dependabot PRs.

When Dependabot bumps a dependency, pyproject.toml gets updated but the
pip_packages lists in src/ogx/providers/registry/*.py can also carry >= floors
that should stay in sync.  This script scans those files and bumps any matching
>= constraint to the new version.

Exit code 0 always. Outputs:
  - One line per file/match explaining what happened.
  - A final "updated=true" or "updated=false" line (for CI consumption).
  - If any files were changed, a "changed_files=<path1>,<path2>,..." line.
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


def update_pip_packages_line(line: str, pkg_name: str, new_version: str) -> tuple[str, bool, str]:
    """Update the >= floor for a package inside a pip_packages string literal.

    Matches entries like "pypdf>=6.7.2" or "mcp>=1.23.0,<2.0" inside Python
    string literals on the given line.

    Returns (new_line, changed, reason).
    """
    pkg_pattern = normalize_pkg_pattern(pkg_name)
    lower_bound_re = re.compile(rf'("(?:{pkg_pattern})(?:\[[^\]]*\])?>=)([\d]+(?:\.[\d]+)*)', re.IGNORECASE)

    match = lower_bound_re.search(line)
    if not match:
        return line, False, f"no >= lower bound for {pkg_name} on this line"

    old_version = match.group(2)
    if parse_version(new_version) <= parse_version(old_version):
        return line, False, f"{pkg_name}: new version {new_version} <= current floor {old_version}"

    upper_bound_re = re.compile(rf'"(?:{pkg_pattern})(?:\[[^\]]*\])?>=(?:[^"]*),<([\d]+(?:\.[\d]+)*)"', re.IGNORECASE)
    upper_match = upper_bound_re.search(line)
    if upper_match:
        upper_version = upper_match.group(1)
        if parse_version(new_version) >= parse_version(upper_version):
            return line, False, (f"{pkg_name}: new version {new_version} >= upper bound <{upper_version}, skipping")

    new_line = lower_bound_re.sub(lambda m: m.group(1) + new_version, line)
    return new_line, True, f"{pkg_name}: updated >= floor from {old_version} to {new_version}"


def find_matching_lines(lines: list[str], pkg_name: str) -> list[int]:
    """Find line indices containing a pip_packages string literal for the given package with a >= floor."""
    pkg_pattern = normalize_pkg_pattern(pkg_name)
    pattern = re.compile(rf'"(?:{pkg_pattern})(?:\[[^\]]*\])?>=', re.IGNORECASE)
    return [i for i, line in enumerate(lines) if pattern.search(line)]


def update_file(filepath: Path, pkg_name: str, new_version: str) -> tuple[bool, list[str]]:
    """Update all pip_packages >= floors for pkg_name in a single file.

    Returns (changed, messages).
    """
    content = filepath.read_text()
    lines = content.splitlines(keepends=True)
    matching = find_matching_lines(lines, pkg_name)

    if not matching:
        return False, []

    changed = False
    messages = []
    for idx in matching:
        new_line, did_change, reason = update_pip_packages_line(lines[idx], pkg_name, new_version)
        if did_change:
            lines[idx] = new_line
            changed = True
            messages.append(f"UPDATED {filepath}: {reason}")
        else:
            messages.append(f"SKIP {filepath}: {reason}")

    if changed:
        filepath.write_text("".join(lines))

    return changed, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Update pip_packages version floors in provider registry files")
    parser.add_argument("--dependency-name", required=True)
    parser.add_argument("--dependency-version", required=True)
    parser.add_argument(
        "--registry-dir",
        default="src/ogx/providers/registry",
        help="Path to the provider registry directory (default: src/ogx/providers/registry)",
    )
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir)
    if not registry_dir.is_dir():
        print(f"Registry directory not found: {registry_dir}", file=sys.stderr)
        print("updated=false")
        return 0

    changed_files: list[str] = []

    for py_file in sorted(registry_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        changed, messages = update_file(py_file, args.dependency_name, args.dependency_version)
        for msg in messages:
            print(msg)
        if changed:
            changed_files.append(str(py_file))

    if changed_files:
        print("updated=true")
        print(f"changed_files={','.join(changed_files)}")
    else:
        print("updated=false")

    return 0


if __name__ == "__main__":
    sys.exit(main())
