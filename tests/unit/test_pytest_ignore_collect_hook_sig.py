# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression for #6428: pytest_ignore_collect must use pytest 9's collection_path."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from types import ModuleType

import pytest


CONFTEST = Path(__file__).resolve().parents[1] / "integration" / "conftest.py"


def _ignore_collect_arg_names() -> list[str]:
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "pytest_ignore_collect":
            return [arg.arg for arg in node.args.args]
    raise AssertionError(f"pytest_ignore_collect not found in {CONFTEST}")


def test_pytest_ignore_collect_uses_collection_path_not_legacy_path() -> None:
    args = _ignore_collect_arg_names()
    assert "collection_path" in args, args
    assert "path" not in args, args


def test_pytest_ignore_collect_signature_registers_under_pytest9() -> None:
    """Fail the same way pytest 9 does when the live hook still uses `path`."""
    args = _ignore_collect_arg_names()
    if "collection_path" in args and "path" not in args:
        src = textwrap.dedent(
            """
            from pathlib import Path
            import pytest

            def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
                return False
            """
        )
    else:
        src = textwrap.dedent(
            """
            import pytest

            def pytest_ignore_collect(path: str, config: pytest.Config) -> bool:
                return False
            """
        )

    mod = ModuleType("ogx_ignore_collect_sig_probe")
    exec(compile(src, "<ogx_ignore_collect_sig_probe>", "exec"), mod.__dict__)
    pm = pytest.PytestPluginManager()
    # PytestPluginManager.register validates hookimpl against hookspec
    pm.register(mod)
