# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib.util
import pathlib

import pytest

_script_path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_target_models_docs.py"
_spec = importlib.util.spec_from_file_location("generate_target_models_docs", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_registry_provider_validation_allows_intentional_exclusions(monkeypatch):
    monkeypatch.setattr(
        _mod,
        "_load_registry_providers",
        lambda: {"openai", "azure", "ollama", "llama-openai-compat", "runpod"},
    )
    monkeypatch.setattr(_mod, "INTENTIONALLY_UNMAPPED_REGISTRY_PROVIDERS", {"runpod"})
    monkeypatch.setattr(
        _mod,
        "SETUP_PROVIDER_ALIASES",
        {
            "gpt": "openai",
            "azure": "azure",
            "ollama": "ollama",
            "llama-api": "llama-openai-compat",
        },
    )

    _mod._validate_registry_provider_coverage()


def test_registry_provider_validation_fails_on_new_unmapped_provider(monkeypatch):
    monkeypatch.setattr(
        _mod,
        "_load_registry_providers",
        lambda: {"openai", "azure", "new-provider"},
    )
    monkeypatch.setattr(_mod, "INTENTIONALLY_UNMAPPED_REGISTRY_PROVIDERS", set())
    monkeypatch.setattr(
        _mod,
        "SETUP_PROVIDER_ALIASES",
        {
            "gpt": "openai",
            "azure": "azure",
        },
    )

    with pytest.raises(ValueError, match="missing registry adapters: `new-provider`"):
        _mod._validate_registry_provider_coverage()


def test_registry_provider_validation_allows_stale_setup_aliases(monkeypatch):
    monkeypatch.setattr(
        _mod,
        "_load_registry_providers",
        lambda: {"openai", "azure"},
    )
    monkeypatch.setattr(_mod, "INTENTIONALLY_UNMAPPED_REGISTRY_PROVIDERS", set())
    monkeypatch.setattr(
        _mod,
        "SETUP_PROVIDER_ALIASES",
        {
            "gpt": "openai",
            "azure": "azure",
            "legacy-tgi": "tgi",
        },
    )

    _mod._validate_registry_provider_coverage()
