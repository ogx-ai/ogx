# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.util
import pathlib
import subprocess
import textwrap

_script_path = pathlib.Path(__file__).resolve().parents[2] / ".github" / "scripts" / "update_registry_deps.py"
_spec = importlib.util.spec_from_file_location("update_registry_deps", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

update_pip_packages_line = _mod.update_pip_packages_line
find_matching_lines = _mod.find_matching_lines
update_file = _mod.update_file


SAMPLE_REGISTRY = textwrap.dedent("""\
    from ogx_api import InlineProviderSpec, RemoteProviderSpec

    def available_providers():
        return [
            InlineProviderSpec(
                provider_type="inline::auto",
                pip_packages=["chardet", "pypdf>=6.7.2", "markitdown[all]"],
            ),
            InlineProviderSpec(
                provider_type="inline::pypdf",
                pip_packages=["chardet", "pypdf>=6.7.2"],
            ),
            InlineProviderSpec(
                provider_type="inline::markitdown",
                pip_packages=["markitdown[all]"],
            ),
            InlineProviderSpec(
                provider_type="inline::mcp",
                pip_packages=["mcp>=1.23.0,<2.0"],
            ),
        ]
""")


class TestUpdatePipPackagesLine:
    def test_updates_lower_bound(self):
        line = '            pip_packages=["chardet", "pypdf>=6.7.2", "markitdown[all]"],\n'
        new_line, changed, reason = update_pip_packages_line(line, "pypdf", "6.13.3")
        assert changed is True
        assert "pypdf>=6.13.3" in new_line
        assert "chardet" in new_line
        assert "markitdown[all]" in new_line

    def test_skips_when_version_not_newer(self):
        line = '            pip_packages=["pypdf>=6.13.3"],\n'
        _, changed, reason = update_pip_packages_line(line, "pypdf", "6.13.3")
        assert changed is False
        assert "<= current floor" in reason

    def test_skips_when_version_lower(self):
        line = '            pip_packages=["pypdf>=6.13.3"],\n'
        _, changed, reason = update_pip_packages_line(line, "pypdf", "6.7.2")
        assert changed is False
        assert "<= current floor" in reason

    def test_skips_when_exceeds_upper_bound(self):
        line = '            pip_packages=["mcp>=1.23.0,<2.0"],\n'
        _, changed, reason = update_pip_packages_line(line, "mcp", "2.0.0")
        assert changed is False
        assert ">= upper bound" in reason

    def test_allows_below_upper_bound(self):
        line = '            pip_packages=["mcp>=1.23.0,<2.0"],\n'
        new_line, changed, _ = update_pip_packages_line(line, "mcp", "1.25.0")
        assert changed is True
        assert "mcp>=1.25.0,<2.0" in new_line

    def test_handles_extras(self):
        line = '            pip_packages=["markitdown[all]>=1.0.0"],\n'
        new_line, changed, _ = update_pip_packages_line(line, "markitdown", "1.2.0")
        assert changed is True
        assert "markitdown[all]>=1.2.0" in new_line

    def test_no_match_returns_unchanged(self):
        line = '            pip_packages=["chardet", "markitdown[all]"],\n'
        _, changed, reason = update_pip_packages_line(line, "pypdf", "6.13.3")
        assert changed is False
        assert "no >= lower bound" in reason

    def test_handles_pep503_normalization(self):
        line = '            pip_packages=["python-socketio>=5.14.0"],\n'
        new_line, changed, _ = update_pip_packages_line(line, "python_socketio", "5.15.0")
        assert changed is True
        assert "python-socketio>=5.15.0" in new_line


class TestFindMatchingLines:
    def test_finds_matching_lines(self):
        lines = SAMPLE_REGISTRY.splitlines(keepends=True)
        indices = find_matching_lines(lines, "pypdf")
        assert len(indices) == 2

    def test_returns_empty_for_no_match(self):
        lines = SAMPLE_REGISTRY.splitlines(keepends=True)
        indices = find_matching_lines(lines, "nonexistent")
        assert len(indices) == 0

    def test_finds_with_upper_bound(self):
        lines = SAMPLE_REGISTRY.splitlines(keepends=True)
        indices = find_matching_lines(lines, "mcp")
        assert len(indices) == 1


class TestUpdateFile:
    def test_updates_multiple_occurrences(self, tmp_path):
        registry_file = tmp_path / "file_processors.py"
        registry_file.write_text(SAMPLE_REGISTRY)

        changed, messages = update_file(registry_file, "pypdf", "6.13.3")
        assert changed is True
        assert len([m for m in messages if "UPDATED" in m]) == 2

        content = registry_file.read_text()
        assert content.count("pypdf>=6.13.3") == 2
        assert "pypdf>=6.7.2" not in content

    def test_no_change_when_not_newer(self, tmp_path):
        registry_file = tmp_path / "file_processors.py"
        registry_file.write_text(SAMPLE_REGISTRY)
        original = registry_file.read_text()

        changed, messages = update_file(registry_file, "pypdf", "6.5.0")
        assert changed is False
        assert registry_file.read_text() == original

    def test_no_match_returns_empty(self, tmp_path):
        registry_file = tmp_path / "file_processors.py"
        registry_file.write_text(SAMPLE_REGISTRY)

        changed, messages = update_file(registry_file, "nonexistent", "1.0.0")
        assert changed is False
        assert messages == []


MULTILINE_REGISTRY = textwrap.dedent("""\
    from ogx_api import InlineProviderSpec

    def available_providers():
        return [
            InlineProviderSpec(
                provider_type="inline::openai",
                pip_packages=[
                    "openai>=1.66.3",
                    "httpx",
                    "aiohttp",
                ],
            ),
        ]
""")


class TestMultiLinePipPackages:
    def test_updates_in_multiline_list(self, tmp_path):
        registry_file = tmp_path / "inference.py"
        registry_file.write_text(MULTILINE_REGISTRY)

        changed, messages = update_file(registry_file, "openai", "1.70.0")
        assert changed is True

        content = registry_file.read_text()
        assert "openai>=1.70.0" in content
        assert "openai>=1.66.3" not in content


class TestMainCli:
    def test_updates_registry_files(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        (registry_dir / "file_processors.py").write_text(SAMPLE_REGISTRY)
        (registry_dir / "__init__.py").write_text("")

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "pypdf",
                "--dependency-version",
                "6.13.3",
                "--registry-dir",
                str(registry_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=true" in result.stdout
        assert "changed_files=" in result.stdout

        content = (registry_dir / "file_processors.py").read_text()
        assert content.count("pypdf>=6.13.3") == 2

    def test_skips_init_file(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        (registry_dir / "__init__.py").write_text('pip_packages=["pypdf>=6.7.2"]')

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "pypdf",
                "--dependency-version",
                "6.13.3",
                "--registry-dir",
                str(registry_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout

    def test_no_change_when_not_found(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()

        (registry_dir / "file_processors.py").write_text(SAMPLE_REGISTRY)

        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "nonexistent-pkg",
                "--dependency-version",
                "1.0.0",
                "--registry-dir",
                str(registry_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout

    def test_missing_registry_dir(self, tmp_path):
        result = subprocess.run(
            [
                "python3",
                str(_script_path),
                "--dependency-name",
                "pypdf",
                "--dependency-version",
                "6.13.3",
                "--registry-dir",
                str(tmp_path / "nonexistent"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "updated=false" in result.stdout
