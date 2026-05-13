# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# The script lives outside the Python package tree, so import its functions
# by path manipulation.
import importlib.util
import pathlib
import subprocess
import textwrap

_script_path = pathlib.Path(__file__).resolve().parents[2] / ".github" / "scripts" / "update_constraint_deps.py"
_spec = importlib.util.spec_from_file_location("update_constraint_deps", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

parse_version = _mod.parse_version
normalize_pkg_pattern = _mod.normalize_pkg_pattern
find_constraint_section = _mod.find_constraint_section
find_constraint_line = _mod.find_constraint_line
update_constraint = _mod.update_constraint
main = _mod.main


SAMPLE_PYPROJECT = textwrap.dedent("""\
    [build-system]
    requires = ["setuptools>=61.0"]

    [tool.uv]
    required-version = ">=0.7.0"
    constraint-dependencies = [
        "aiohttp>=3.13.4",                # CVE-2026-34514
        "litellm<1.83.7",                  # upper-bound only
        "pydantic>=2.11.9,<2.12.0",        # pinned range
        "urllib3>=2.6.3",
        "transformers>=4.57.2,<5.0.0",     # wide upper bound
    ]

    [project]
    name = "example"
    dependencies = [
        "requests>=2.28.0",
        "pydantic>=2.11.9",
    ]
""")


class TestParseVersion:
    def test_simple_version(self):
        assert parse_version("1.2.3") == (1, 2, 3)

    def test_two_part_version(self):
        assert parse_version("2.0") == (2, 0)

    def test_four_part_version(self):
        assert parse_version("1.2.3.4") == (1, 2, 3, 4)

    def test_comparison(self):
        assert parse_version("2.12.5") >= parse_version("2.12.0")
        assert parse_version("3.13.2") <= parse_version("3.13.4")
        assert parse_version("1.0.0") < parse_version("1.0.1")


class TestNormalizePkgPattern:
    def test_hyphens_become_character_class(self):
        pattern = normalize_pkg_pattern("python-multipart")
        assert "[-_.]" in pattern

    def test_underscores_become_character_class(self):
        pattern = normalize_pkg_pattern("python_socketio")
        assert "[-_.]" in pattern

    def test_dots_become_character_class(self):
        pattern = normalize_pkg_pattern("zope.interface")
        assert "[-_.]" in pattern

    def test_lowercased(self):
        pattern = normalize_pkg_pattern("PyYAML")
        assert pattern == "pyyaml"


class TestFindConstraintSection:
    def test_finds_section(self):
        lines = SAMPLE_PYPROJECT.splitlines(keepends=True)
        result = find_constraint_section(lines)
        assert result is not None
        start, end = result
        assert "constraint-dependencies" in lines[start]
        assert lines[end].rstrip().endswith("]")

    def test_excludes_project_dependencies(self):
        """The section finder must not confuse [project] dependencies with constraint-dependencies."""
        lines = SAMPLE_PYPROJECT.splitlines(keepends=True)
        start, end = find_constraint_section(lines)
        section_text = "".join(lines[start : end + 1])
        assert "requests" not in section_text

    def test_returns_none_when_missing(self):
        lines = textwrap.dedent("""\
            [project]
            name = "no-constraints"
            dependencies = ["requests"]
        """).splitlines(keepends=True)
        assert find_constraint_section(lines) is None


class TestFindConstraintLine:
    def test_finds_exact_match(self):
        lines = SAMPLE_PYPROJECT.splitlines(keepends=True)
        idx = find_constraint_line(lines, "aiohttp")
        assert idx is not None
        assert "aiohttp" in lines[idx]

    def test_finds_with_hyphen_underscore_equivalence(self):
        """PEP 503: python-socketio and python_socketio are equivalent."""
        pyproject = textwrap.dedent("""\
            [tool.uv]
            constraint-dependencies = [
                "python-socketio>=5.14.0",
            ]
        """)
        lines = pyproject.splitlines(keepends=True)
        assert find_constraint_line(lines, "python_socketio") is not None
        assert find_constraint_line(lines, "python-socketio") is not None

    def test_does_not_match_outside_constraints(self):
        """Packages in [project] dependencies must not be found."""
        lines = SAMPLE_PYPROJECT.splitlines(keepends=True)
        assert find_constraint_line(lines, "requests") is None

    def test_returns_none_for_unknown_package(self):
        lines = SAMPLE_PYPROJECT.splitlines(keepends=True)
        assert find_constraint_line(lines, "nonexistent-pkg") is None


class TestUpdateConstraint:
    def test_updates_lower_bound(self):
        line = '    "aiohttp>=3.13.4",                # CVE-2026-34514\n'
        new_line, changed, reason = update_constraint(line, "aiohttp", "3.14.0")
        assert changed is True
        assert ">=3.14.0" in new_line
        assert "CVE-2026-34514" in new_line

    def test_preserves_upper_bound(self):
        line = '    "transformers>=4.57.2,<5.0.0",     # comment\n'
        new_line, changed, _ = update_constraint(line, "transformers", "4.58.0")
        assert changed is True
        assert ">=4.58.0,<5.0.0" in new_line

    def test_skips_when_no_lower_bound(self):
        line = '    "litellm<1.83.7",                  # upper-bound only\n'
        _, changed, reason = update_constraint(line, "litellm", "1.84.0")
        assert changed is False
        assert "no >= lower bound" in reason

    def test_skips_when_version_not_newer(self):
        line = '    "aiohttp>=3.13.4",\n'
        _, changed, reason = update_constraint(line, "aiohttp", "3.13.4")
        assert changed is False
        assert "<= current floor" in reason

    def test_skips_when_version_lower(self):
        line = '    "aiohttp>=3.13.4",\n'
        _, changed, reason = update_constraint(line, "aiohttp", "3.13.2")
        assert changed is False
        assert "<= current floor" in reason

    def test_skips_when_new_version_exceeds_upper_bound(self):
        line = '    "pydantic>=2.11.9,<2.12.0",        # pinned range\n'
        _, changed, reason = update_constraint(line, "pydantic", "2.12.5")
        assert changed is False
        assert ">= upper bound" in reason

    def test_allows_version_below_upper_bound(self):
        line = '    "pydantic>=2.11.9,<2.12.0",        # pinned range\n'
        new_line, changed, _ = update_constraint(line, "pydantic", "2.11.10")
        assert changed is True
        assert ">=2.11.10,<2.12.0" in new_line

    def test_preserves_whitespace_and_comment(self):
        line = '    "urllib3>=2.6.3",\n'
        new_line, changed, _ = update_constraint(line, "urllib3", "2.7.0")
        assert changed is True
        assert new_line.startswith("    ")
        assert new_line.endswith(",\n")

    def test_handles_pep503_name_normalization(self):
        line = '    "python-socketio>=5.14.0",         # CVE\n'
        new_line, changed, _ = update_constraint(line, "python_socketio", "5.15.0")
        assert changed is True
        assert "python-socketio>=5.15.0" in new_line


class TestMainCli:
    """End-to-end tests for the CLI entry point."""

    def test_updates_constraint_in_file(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(SAMPLE_PYPROJECT)

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "aiohttp",
                "--dependency-version",
                "3.14.0",
                "--pyproject",
                str(pyproject),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=true" in result.stdout

        content = pyproject.read_text()
        assert "aiohttp>=3.14.0" in content
        assert "CVE-2026-34514" in content

    def test_skips_unknown_package(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(SAMPLE_PYPROJECT)

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "nonexistent",
                "--dependency-version",
                "1.0.0",
                "--pyproject",
                str(pyproject),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout
        assert pyproject.read_text() == SAMPLE_PYPROJECT

    def test_skips_upper_bound_conflict(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(SAMPLE_PYPROJECT)

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "pydantic",
                "--dependency-version",
                "2.12.5",
                "--pyproject",
                str(pyproject),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout
        assert "pydantic>=2.11.9,<2.12.0" in pyproject.read_text()

    def test_does_not_modify_project_dependencies(self, tmp_path):
        """Updating a package that exists in [project] dependencies but not in
        constraint-dependencies must not touch pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(SAMPLE_PYPROJECT)

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "requests",
                "--dependency-version",
                "2.32.0",
                "--pyproject",
                str(pyproject),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout
        assert pyproject.read_text() == SAMPLE_PYPROJECT

    def test_missing_pyproject_returns_error(self, tmp_path):
        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "aiohttp",
                "--dependency-version",
                "3.14.0",
                "--pyproject",
                str(tmp_path / "nope.toml"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_file_unchanged_when_version_not_newer(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(SAMPLE_PYPROJECT)
        original = pyproject.read_text()

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "aiohttp",
                "--dependency-version",
                "3.13.2",
                "--pyproject",
                str(pyproject),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout
        assert pyproject.read_text() == original
