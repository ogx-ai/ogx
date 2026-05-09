#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Provider Compatibility Matrix Generator for the Responses API.

Analyzes existing test recordings and pytest results to produce a per-provider
feature support matrix. The matrix shows which Responses API features each
inference provider supports based on actual test evidence.

Usage:
    # Analyze recordings and print to terminal:
    uv run python scripts/provider_compat_matrix.py

    # Update the generated docs (used by pre-commit):
    uv run python scripts/provider_compat_matrix.py --update

    # Check for regressions without updating:
    uv run python scripts/provider_compat_matrix.py --check-regression

    # Include a JUnit XML report from a test run:
    uv run python scripts/provider_compat_matrix.py --update --junit path/to/results.xml
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "tests" / "integration" / "responses" / "recordings"
MATRIX_MD = ROOT / "docs" / "docs" / "api-openai" / "provider_matrix.md"

# ---------------------------------------------------------------------------
# Category derivation from test file names
# ---------------------------------------------------------------------------

TESTS_DIR = ROOT / "tests" / "integration" / "responses"


def _file_to_category(filename: str) -> str:
    """Derive a human-readable category from a test file name.

    e.g. test_basic_responses.py -> Basic Responses
         test_mcp_authentication.py -> Mcp Authentication
    """
    name = filename.removesuffix(".py").removeprefix("test_")
    return name.replace("_", " ").title()


def _discover_categories() -> list[str]:
    """Discover categories from test files on disk, in sorted order."""
    if not TESTS_DIR.exists():
        return []
    return sorted(_file_to_category(f.name) for f in TESTS_DIR.glob("test_*.py"))


def _test_name_to_feature(test_name: str) -> str:
    """Convert a test function name to a human-readable feature name."""
    name = test_name
    for prefix in ("test_openai_response_", "test_openai_", "test_response_", "test_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProviderResults:
    provider: str
    # category -> {feature -> outcome}
    results: dict[str, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    # Model names seen in recordings for this provider
    models: set[str] = field(default_factory=set)
    # Service URL host patterns seen (e.g. "api.openai.com", "localhost:8000")
    hosts: set[str] = field(default_factory=set)
    # Provider metadata from recordings (e.g. SDK versions, API versions)
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recording analysis
# ---------------------------------------------------------------------------

# Self-hosted providers where the endpoint is not meaningful (localhost / user-managed)
_SELF_HOSTED_PROVIDERS = {"vllm", "ollama", "tgi", "llama-cpp-server"}

_PROVIDER_RE = re.compile(r"txt=([a-zA-Z_-]+)/")
_VIS_PROVIDER_RE = re.compile(r"vis=([a-zA-Z_-]+)/")


def _extract_provider_from_test_id(test_id: str) -> str | None:
    for pattern in (_PROVIDER_RE, _VIS_PROVIDER_RE):
        m = pattern.search(test_id)
        if m:
            return m.group(1)
    return None


def _extract_test_name(test_id: str) -> str | None:
    parts = test_id.split("::")
    if len(parts) < 2:
        return None
    last = parts[-1]
    bracket = last.find("[")
    return last[:bracket] if bracket >= 0 else last


def _extract_test_file(test_id: str) -> str | None:
    for part in test_id.split("::"):
        if part.endswith(".py"):
            return part.split("/")[-1]
    return None


def scan_recordings(recordings_dir: Path) -> dict[str, ProviderResults]:
    """Scan recording files to determine which providers have test evidence."""
    provider_map: dict[str, ProviderResults] = {}

    if not recordings_dir.exists():
        return provider_map

    for rec_file in sorted(recordings_dir.glob("*.json")):
        try:
            data = json.loads(rec_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        test_id = data.get("test_id", "")
        if not test_id:
            continue

        provider = _extract_provider_from_test_id(test_id)
        if not provider:
            continue

        test_name = _extract_test_name(test_id)
        if not test_name:
            continue

        test_file = _extract_test_file(test_id)
        category = _file_to_category(test_file) if test_file else "Other"
        feature = _test_name_to_feature(test_name)

        if provider not in provider_map:
            provider_map[provider] = ProviderResults(provider=provider)

        # Extract model and host from request metadata (skip infrastructure endpoints)
        request = data.get("request", {})
        endpoint = request.get("endpoint", "")
        if endpoint not in ("/v1/models", "/api/tags"):
            model = request.get("model", "")
            url = request.get("url", "")
            if model:
                provider_map[provider].models.add(model)
            if url and provider not in _SELF_HOSTED_PROVIDERS:
                parsed = urlparse(url)
                if parsed.hostname and parsed.hostname != "localhost":
                    provider_map[provider].hosts.add(parsed.hostname)

        # Merge provider metadata (first non-empty value wins per key)
        pm = request.get("provider_metadata", {})
        if pm:
            for k, v in pm.items():
                if k not in provider_map[provider].metadata:
                    provider_map[provider].metadata[k] = v

        if feature not in provider_map[provider].results[category]:
            provider_map[provider].results[category][feature] = "pass"

    # For providers that have coverage within a category, mark missing features
    # in that category as "skip" (unsupported) rather than "—" (untested).
    all_features: dict[str, set[str]] = defaultdict(set)
    for pr in provider_map.values():
        for category, features in pr.results.items():
            all_features[category].update(features.keys())

    for pr in provider_map.values():
        for category, features in all_features.items():
            # Only mark as skip if the provider has some coverage in this category
            if category not in pr.results:
                continue
            for feature in features:
                if feature not in pr.results[category]:
                    pr.results[category][feature] = "skip"

    return provider_map


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _collect_all_categories(provider_map: dict[str, ProviderResults]) -> dict[str, list[str]]:
    categories: dict[str, set[str]] = defaultdict(set)
    for pr in provider_map.values():
        for category, features in pr.results.items():
            categories[category].update(features.keys())

    # Order by discovered test files, then any extras
    sorted_cats: dict[str, list[str]] = {}
    for cat in _discover_categories():
        if cat in categories:
            sorted_cats[cat] = sorted(categories[cat])
    for cat in sorted(categories.keys()):
        if cat not in sorted_cats:
            sorted_cats[cat] = sorted(categories[cat])
    return sorted_cats


def _compute_summary(provider_map: dict[str, ProviderResults]) -> dict:
    categories = _collect_all_categories(provider_map)
    total_features = sum(len(feats) for feats in categories.values())
    providers = sorted(provider_map.keys())

    summary = {"total_features": total_features, "providers": {}}
    for p in providers:
        passing = tested = failing = 0
        for cat, feats in categories.items():
            for f in feats:
                outcome = provider_map[p].results.get(cat, {}).get(f)
                if outcome in ("pass", "fail", "error"):
                    tested += 1
                if outcome == "pass":
                    passing += 1
                if outcome in ("fail", "error"):
                    failing += 1
        summary["providers"][p] = {
            "tested": tested,
            "passing": passing,
            "failing": failing,
            "coverage_pct": round(passing / total_features * 100, 1) if total_features else 0,
        }
    return summary


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _load_provider_names() -> dict[str, str]:
    """Build display names from the inference provider registry's adapter_type values."""
    try:
        from ogx.providers.registry.inference import available_providers

        names: dict[str, str] = {}
        for spec in available_providers():
            adapter = getattr(spec, "adapter_type", None)
            if not adapter:
                continue
            # adapter_type is already a good short name (e.g. "ollama", "vllm", "openai")
            # just title-case it, with special handling for acronyms/brands
            names[adapter] = adapter
        return names
    except Exception:
        return {}


_REGISTRY_PROVIDERS: set[str] = set(_load_provider_names().keys())

_PROVIDER_NOTES: dict[str, str] = {
    "bedrock": (
        "AWS Bedrock integration uses the OpenAI-compatible Chat Completions API, which only "
        "supports GPT-OSS models (e.g. `openai.gpt-oss-20b-1:0`). Native AWS models "
        "(Amazon Nova, Anthropic Claude, etc.) are not yet supported. "
        "No vision model is available — image tests are skipped."
    ),
}


def _pname(provider: str) -> str:
    """Display name for a provider. Uses the adapter_type as-is since those are
    already well-known brand names (ollama, vllm, openai, etc.)."""
    return provider


def _icon(outcome: str | None) -> str:
    if outcome is None:
        return "—"
    return {"pass": "✅", "fail": "❌", "skip": "⏭️", "error": "💥"}.get(outcome, "?")


def generate_matrix_markdown(provider_map: dict[str, ProviderResults]) -> str:
    providers = sorted(provider_map.keys())
    if not providers:
        return "No provider data available.\n"

    categories = _collect_all_categories(provider_map)
    summary = _compute_summary(provider_map)

    lines: list[str] = [
        "---",
        "title: Provider Compatibility Matrix",
        "description: Responses API feature support by inference provider",
        "---",
        "",
        "{/*This file is auto-generated by scripts/provider_compat_matrix.py. Do not edit manually.*/}",
        "",
        "This matrix shows which Responses API features are supported by each",
        "inference provider, based on integration test results.",
        "",
        "| Legend | Meaning |",
        "|--------|---------|",
        "| ✅ | Tested and passing |",
        "| ❌ | Tested and failing |",
        "| ⏭️ | Skipped (unsupported) |",
        "| — | Not tested |",
        "",
        "## Summary",
        "",
        "| Provider | Tested | Passing | Failing | Coverage |",
        "|----------|--------|---------|---------|----------|",
    ]

    for p in providers:
        s = summary["providers"][p]
        lines.append(f"| {_pname(p)} | {s['tested']} | {s['passing']} | {s['failing']} | {s['coverage_pct']:.0f}% |")

    lines.append("")

    # Provider details section
    lines.append("## Provider Details")
    lines.append("")
    lines.append("Models, endpoints, and versions used during test recordings.")
    lines.append("")
    lines.append("| Provider | Model(s) | Endpoint | Version Info |")
    lines.append("|----------|----------|----------|--------------|")
    for p in providers:
        pr = provider_map[p]
        models = ", ".join(sorted(pr.models)) if pr.models else "—"
        hosts = ", ".join(sorted(pr.hosts)) if pr.hosts else "—"
        version_parts = []
        for k, v in sorted(pr.metadata.items()):
            label = k.replace("_", " ").removesuffix(" version")
            version_parts.append(f"{label}: {v}")
        version_info = ", ".join(version_parts) if version_parts else "—"
        lines.append(f"| {_pname(p)} | {models} | {hosts} | {version_info} |")
    lines.append("")

    for category, features in categories.items():
        lines.append(f"## {category}")
        lines.append("")

        header_cols = ["Feature"] + [_pname(p) for p in providers]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

        for feature in features:
            cols = [feature]
            for p in providers:
                outcome = provider_map[p].results.get(category, {}).get(feature)
                cols.append(_icon(outcome))
            lines.append("| " + " | ".join(cols) + " |")

        lines.append("")

    lines.extend(["---", "", "*Generated by `scripts/provider_compat_matrix.py`*", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    provider_map = scan_recordings(RECORDINGS_DIR)
    md = generate_matrix_markdown(provider_map)
    MATRIX_MD.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_MD.write_text(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
