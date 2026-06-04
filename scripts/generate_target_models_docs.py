#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUIRTES_PY = ROOT / "tests" / "integration" / "suites.py"
CI_MATRIX_JSON = ROOT / "tests" / "integration" / "ci_matrix.json"
TARGET_MODELS_MD = ROOT / "tests" / "integration" / "TARGET_MODELS.md"

sys.path.insert(0, str(ROOT / "scripts"))

from provider_compat_matrix import RECORDINGS_DIR, scan_recordings

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure": "Azure",
    "bedrock": "Bedrock",
    "vertexai": "Vertex AI",
    "watsonx": "WatsonX",
    "vllm": "vLLM",
    "ollama": "Ollama",
    "gemini": "Gemini",
    "anthropic": "Anthropic",
    "groq": "Groq",
    "fireworks": "Fireworks",
    "databricks": "Databricks",
    "together": "Together",
    "cerebras": "Cerebras",
    "tgi": "TGI",
    "llama-cpp-server": "llama.cpp server",
    "llama-openai-compat": "Llama API",
}

SETUP_PROVIDER_ALIASES = {
    "gpt": "openai",
    "gpt-reasoning": "openai",
    "azure": "azure",
    "bedrock": "bedrock",
    "vertexai": "vertexai",
    "watsonx": "watsonx",
    "vllm": "vllm",
    "vllm-qwen3next": "vllm",
    "ollama": "ollama",
    "ollama-vision": "ollama",
    "ollama-reasoning": "ollama",
    "gemini": "gemini",
    "anthropic": "anthropic",
    "groq": "groq",
    "fireworks": "fireworks",
    "databricks": "databricks",
    "together": "together",
    "cerebras": "cerebras",
    "tgi": "tgi",
    "llama-cpp-server": "llama-cpp-server",
    "llama-api": "llama-openai-compat",
}

# These inference providers exist in the registry but do not have named
# integration-test setups yet, so they are intentionally excluded from this doc.
INTENTIONALLY_UNMAPPED_REGISTRY_PROVIDERS = {
    "nvidia",
    "oci",
    "passthrough",
    "runpod",
    "sambanova",
}


def _load_suite_data() -> tuple[dict[str, object], dict[str, object]]:
    data = runpy.run_path(str(SUIRTES_PY))
    return data["SETUP_DEFINITIONS"], data["SUITE_DEFINITIONS"]


def _load_ci_matrix() -> dict[str, object]:
    return json.loads(CI_MATRIX_JSON.read_text())


def _load_registry_providers() -> set[str]:
    from ogx.providers.registry.inference import available_providers

    return {spec.adapter_type for spec in available_providers() if getattr(spec, "adapter_type", None)}


def _validate_registry_provider_coverage() -> None:
    registry_providers = _load_registry_providers()
    mapped_providers = set(SETUP_PROVIDER_ALIASES.values())

    missing_providers = sorted(registry_providers - mapped_providers - INTENTIONALLY_UNMAPPED_REGISTRY_PROVIDERS)
    if missing_providers:
        raise ValueError(
            "Target model matrix provider mapping is out of sync with the inference registry, "
            + "missing registry adapters: "
            + ", ".join(f"`{provider}`" for provider in missing_providers)
            + ". Update SETUP_PROVIDER_ALIASES or INTENTIONALLY_UNMAPPED_REGISTRY_PROVIDERS."
        )


def _collect_all_categories(provider_map: dict[str, object]) -> dict[str, list[str]]:
    categories: dict[str, set[str]] = defaultdict(set)
    for provider_results in provider_map.values():
        for category, features in provider_results.results.items():
            categories[category].update(features.keys())
    return {category: sorted(features) for category, features in sorted(categories.items())}


def _responses_summary() -> tuple[int, dict[str, dict[str, int | float]]]:
    provider_map = scan_recordings(RECORDINGS_DIR)
    categories = _collect_all_categories(provider_map)
    total_features = sum(len(features) for features in categories.values())
    summary: dict[str, dict[str, int | float]] = {}

    for provider, provider_results in sorted(provider_map.items()):
        tested = 0
        passing = 0
        for category, features in categories.items():
            for feature in features:
                outcome = provider_results.results.get(category, {}).get(feature)
                if outcome in ("pass", "fail", "error"):
                    tested += 1
                if outcome == "pass":
                    passing += 1

        summary[provider] = {
            "tested": tested,
            "passing": passing,
            "coverage_pct": round((passing / total_features) * 100) if total_features else 0,
        }

    return total_features, summary


def _format_model(value: str | int | None) -> str:
    if value is None:
        return "—"
    return str(value)


def _format_provider(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider)


def _format_suite_list(suites: list[str]) -> str:
    if not suites:
        return "—"
    return ", ".join(f"`{suite}`" for suite in suites)


def _ci_usage_by_setup(ci_matrix: dict[str, object]) -> dict[str, dict[str, list[str]]]:
    usage: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"default": [], "scheduled": []})

    for job in ci_matrix.get("default", []):
        usage[job["setup"]]["default"].append(job["suite"])

    for jobs in ci_matrix.get("schedules", {}).values():
        for job in jobs:
            usage[job["setup"]]["scheduled"].append(job["suite"])

    return usage


def _suite_scope_note(suite_definition: object) -> str | None:
    roots = suite_definition.roots
    if len(roots) == 1:
        leaf = roots[0].split("/")[-1]
        return f"`{leaf.split('::', 1)[0]}` only"
    if len(roots) > 1 and len(roots) <= 6:
        return f"{len(roots)} roots"
    return None


def _job_notes(
    job: dict[str, object],
    suite_definitions: dict[str, object],
    responses_total: int,
    responses_summary: dict[str, dict[str, int | float]],
) -> str:
    notes: list[str] = []

    allowed_clients = job.get("allowed_clients", [])
    if allowed_clients:
        notes.append(f"{', '.join(allowed_clients)} client only")

    stack_config = str(job.get("stack_config", ""))
    if "postgres" in stack_config:
        notes.append("Postgres store")

    suite_definition = suite_definitions.get(job["suite"])
    if suite_definition is not None:
        scope_note = _suite_scope_note(suite_definition)
        if scope_note:
            notes.append(scope_note)

    provider = SETUP_PROVIDER_ALIASES.get(job["setup"])
    coverage = responses_summary.get(provider)
    if coverage and job["suite"] in {
        "responses",
        "bedrock-responses",
        "gpt-reasoning",
        "vllm-reasoning",
        "ollama-reasoning",
    }:
        notes.append(f"Responses coverage: {coverage['passing']}/{responses_total} ({coverage['coverage_pct']}%)")

    return "; ".join(notes)


def _render_ci_jobs(
    jobs: list[dict[str, object]],
    suite_definitions: dict[str, object],
    responses_total: int,
    responses_summary: dict[str, dict[str, int | float]],
) -> list[str]:
    lines = [
        "| Suite | Setup | Notes |",
        "|-------|-------|-------|",
    ]
    for job in jobs:
        notes = _job_notes(job, suite_definitions, responses_total, responses_summary)
        lines.append(f"| `{job['suite']}` | `{job['setup']}` | {notes} |")
    lines.append("")
    return lines


def _render_setup_table(
    title: str,
    setup_names: list[str],
    setup_definitions: dict[str, object],
    ci_usage: dict[str, dict[str, list[str]]],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Setup | Text Model | Vision Model | Embedding Model | Safety Model | Default CI | Scheduled CI |",
        "|-------|------------|--------------|-----------------|--------------|------------|--------------|",
    ]

    for setup_name in setup_names:
        setup = setup_definitions[setup_name]
        defaults = setup.defaults
        usage = ci_usage.get(setup_name, {"default": [], "scheduled": []})
        lines.append(
            "| "
            f"`{setup_name}` | "
            f"{_format_model(defaults.get('text_model'))} | "
            f"{_format_model(defaults.get('vision_model'))} | "
            f"{_format_model(defaults.get('embedding_model'))} | "
            f"{_format_model(defaults.get('safety_model'))} | "
            f"{_format_suite_list(usage['default'])} | "
            f"{_format_suite_list(usage['scheduled'])} |"
        )

    lines.append("")
    return lines


def _render_responses_summary(responses_total: int, responses_summary: dict[str, dict[str, int | float]]) -> list[str]:
    lines = [
        "## Responses Coverage Summary",
        "",
        "This section is derived from the same replay recordings used to generate "
        "`docs/docs/api-openai/provider_matrix.md`.",
        "",
        "| Provider | Tested | Passing | Coverage |",
        "|----------|--------|---------|----------|",
    ]

    for provider, summary in sorted(
        responses_summary.items(),
        key=lambda item: (-int(item[1]["coverage_pct"]), _format_provider(item[0]).lower()),
    ):
        lines.append(
            f"| {_format_provider(provider)} | "
            f"{summary['tested']} | {summary['passing']} | {summary['coverage_pct']}% |"
        )

    lines.extend(
        [
            "",
            f"Total Responses features counted: {responses_total}.",
            "",
        ]
    )
    return lines


def generate_target_models_docs() -> str:
    _validate_registry_provider_coverage()
    setup_definitions, suite_definitions = _load_suite_data()
    ci_matrix = _load_ci_matrix()
    responses_total, responses_summary = _responses_summary()
    ci_usage = _ci_usage_by_setup(ci_matrix)

    ci_backed_setups = sorted(ci_usage.keys())
    additional_setups = sorted(
        setup_name for setup_name in setup_definitions if setup_name not in set(ci_backed_setups)
    )

    lines = [
        "# Target Model Matrix",
        "",
        "<!-- This file is auto-generated by scripts/generate_target_models_docs.py. -->",
        "",
        "This document makes explicit the setups and target models defined in "
        "`tests/integration/suites.py` and the CI lanes configured in "
        "`tests/integration/ci_matrix.json`.",
        "",
        "## CI Lanes (Default)",
        "",
        "These jobs come from the `default` section of `ci_matrix.json`. They all run in the "
        "merge queue, while PR-triggered execution still depends on which files changed.",
        "",
    ]
    lines.extend(_render_ci_jobs(ci_matrix.get("default", []), suite_definitions, responses_total, responses_summary))

    schedules = ci_matrix.get("schedules", {})
    if schedules:
        lines.extend(["## CI Lanes (Scheduled)", ""])
        for cron, jobs in sorted(schedules.items()):
            lines.append(f"Cron: `{cron}`")
            lines.append("")
            lines.extend(_render_ci_jobs(jobs, suite_definitions, responses_total, responses_summary))

    lines.extend(_render_setup_table("CI-backed Setups", ci_backed_setups, setup_definitions, ci_usage))
    lines.extend(_render_setup_table("Additional Named Setups", additional_setups, setup_definitions, ci_usage))
    lines.extend(_render_responses_summary(responses_total, responses_summary))
    return "\n".join(lines)


def main() -> int:
    TARGET_MODELS_MD.write_text(generate_target_models_docs(), encoding="utf-8")
    print(f"Generated {TARGET_MODELS_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
