# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for the inline::sentence-transformers provider dependency declarations.

Without a lower bound on ``sentence-transformers`` the resolver can pick an ancient release
(e.g. 0.2.3) whose ``SentenceTransformer.__init__`` does not accept ``trust_remote_code=``, so
every embeddings request raises ``TypeError``. The floor declared in the provider registry must
match the one in the ``starter`` extra of pyproject.toml so ``ogx stack build`` / ``list-deps`` and
``pip install ogx[starter]`` cannot drift apart.
"""

import pathlib
import tomllib

from packaging.requirements import Requirement

from ogx.core.distribution import get_provider_registry
from ogx.providers.registry.inference import BUILTIN_DEPS
from ogx_api import Api

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
PACKAGE = "sentence-transformers"


def _requirement_for(package: str, entries: list[str]) -> Requirement:
    matches = [Requirement(entry) for entry in entries if Requirement(entry).name.lower() == package]
    assert len(matches) == 1, f"expected exactly one {package} entry in {entries}, found {matches}"
    return matches[0]


def _starter_extra_floor() -> str:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    starter = pyproject["project"]["optional-dependencies"]["starter"]
    specifiers = [s for s in _requirement_for(PACKAGE, starter).specifier if s.operator == ">="]
    assert specifiers, f"{PACKAGE} has no >= floor in the starter extra"
    return specifiers[0].version


def _provider_pip_packages() -> list[str]:
    spec = get_provider_registry()[Api.inference]["inline::sentence-transformers"]
    # skip the "torch ... --extra-index-url ..." entry, which is not a single PEP 508 requirement
    return [entry for entry in spec.pip_packages if "--" not in entry]


def test_provider_pip_packages_declare_sentence_transformers_floor():
    requirement = _requirement_for(PACKAGE, _provider_pip_packages())
    floors = [s.version for s in requirement.specifier if s.operator == ">="]
    assert floors, f"{PACKAGE} must declare a >= lower bound in the provider registry, got {requirement!r}"
    assert floors[0] == _starter_extra_floor()


def test_builtin_deps_declare_sentence_transformers_floor():
    requirement = _requirement_for(PACKAGE, BUILTIN_DEPS)
    floors = [s.version for s in requirement.specifier if s.operator == ">="]
    assert floors, f"{PACKAGE} must declare a >= lower bound in BUILTIN_DEPS, got {requirement!r}"
    assert floors[0] == _starter_extra_floor()
