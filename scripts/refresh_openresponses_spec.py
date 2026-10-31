#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Refresh docs/static/openresponses-spec.json

docs/static/openresponses-spec.json is the reference spec scripts/openai_coverage.py
diffs against for the "Responses" category (see --openresponses-spec there). It is a
hand-picked slice of OpenAI's full spec: just the stable (non-beta) /responses* paths,
plus every schema those paths actually reference, transitively. We keep it separate
from the full OpenAI spec (docs/static/openai-spec-*.yml) because that file has lagged
behind the Responses API specifically, and periodically regenerating just this slice is
far cheaper than re-vendoring the (several-MB) full spec.

Policy: full surface, not a maintained subset. "Full surface" here means every schema
transitively reachable from the stable /responses* paths in OpenAI's own spec --
not a hand-curated list of types we decide matter. Beta paths and schemas (any path
whose key contains "?beta=true" in OpenAI's spec, and any Beta*-prefixed schema) are
excluded on purpose: OGX targets the stable contract, and the stable paths' own schemas
do not reference beta-only types (verified when this script was written -- if a future
OpenAI spec version changes that, this script will surface it as a resolution error
below rather than silently pulling beta types in).

Usage:
    # 1. Fetch a current copy of OpenAI's spec, e.g.:
    curl -sL -o /tmp/openai-openapi.yaml \\
        https://raw.githubusercontent.com/openai/openai-openapi/main/openapi.yaml

    # 2. Regenerate the slice:
    python scripts/refresh_openresponses_spec.py /tmp/openai-openapi.yaml

    # 3. Re-run coverage and accept the resulting score/doc changes:
    python scripts/openai_coverage.py --update --generate-docs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import yaml

# The stable (non-beta) paths tagged "Responses" in OpenAI's spec, as of the version
# this script was last run against. OpenAI represents beta variants of these same
# paths as separate path keys suffixed "?beta=true" (e.g. "/responses?beta=true"),
# which this list intentionally omits.
STABLE_RESPONSES_PATHS = [
    "/responses",
    "/responses/{response_id}",
    "/responses/{response_id}/cancel",
    "/responses/{response_id}/input_items",
    "/responses/input_tokens",
    "/responses/compact",
]

# OpenAPI component sections that can be $ref'd from a path item and can themselves
# $ref other components (e.g. a reusable "responses" entry's content.schema), so the
# closure below must resolve both kinds, not just schemas.
_COMPONENT_REF_RE = re.compile(r"^#/components/(schemas|responses)/(.+)$")


def _load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        return cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], json.loads(text))


def _find_component_refs(obj: Any) -> set[tuple[str, str]]:
    """Recursively collect every (section, name) pair $ref'd from obj, e.g.
    ("schemas", "FunctionToolCall") for "#/components/schemas/FunctionToolCall"."""
    found: set[tuple[str, str]] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str):
                match = _COMPONENT_REF_RE.match(value)
                if match:
                    found.add((match.group(1), match.group(2)))
            else:
                found |= _find_component_refs(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= _find_component_refs(item)
    return found


def _component_closure(seed: Any, components: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Resolve every schemas/responses component transitively referenced from seed,
    failing loudly on a dangling reference rather than silently dropping it."""
    needed = _find_component_refs(seed)
    closure: set[tuple[str, str]] = set()
    while needed - closure:
        newly_needed = needed - closure
        closure |= newly_needed
        for section, name in newly_needed:
            section_components = components.get(section, {})
            if name not in section_components:
                raise KeyError(
                    f"Failed to resolve {section} reference '{name}' while building the "
                    "openresponses-spec.json closure. The source spec may have renamed "
                    "or removed a component that a /responses* path still references."
                )
            needed |= _find_component_refs(section_components[name])

    resolved: dict[str, dict[str, Any]] = {}
    for section, name in closure:
        resolved.setdefault(section, {})[name] = components.get(section, {})[name]
    for section in resolved:
        resolved[section] = dict(sorted(resolved[section].items()))
    return resolved


def refresh(source_spec: dict[str, Any]) -> dict[str, Any]:
    all_paths = source_spec.get("paths", {})
    selected_paths: dict[str, Any] = {}
    for path in STABLE_RESPONSES_PATHS:
        if path not in all_paths:
            raise KeyError(
                f"Failed to find expected path '{path}' in the source spec. OpenAI may "
                "have renamed or removed it -- update STABLE_RESPONSES_PATHS above."
            )
        selected_paths[path] = all_paths[path]

    closure = _component_closure(selected_paths, source_spec.get("components", {}))

    beta_names = sorted(name for section in closure.values() for name in section if name.startswith("Beta"))
    if beta_names:
        raise RuntimeError(
            "The stable /responses* paths now transitively reference beta-only "
            f"components: {beta_names}. This script deliberately excludes beta "
            "surface area -- decide whether to include these intentionally before "
            "proceeding, then update this script's policy comment above."
        )

    source_info = source_spec.get("info", {})
    return {
        "openapi": source_spec.get("openapi", "3.1.0"),
        "info": {
            "title": source_info.get("title", "OpenAI API"),
            "version": source_info.get("version", "unknown"),
        },
        "servers": source_spec.get("servers", [{"url": "https://api.openai.com/v1"}]),
        "paths": selected_paths,
        "components": closure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Path to a local copy of OpenAI's openapi.yaml or .json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/static/openresponses-spec.json"),
        help="Where to write the refreshed spec",
    )
    args = parser.parse_args()

    source_spec = _load_spec(args.source)
    output_spec = refresh(source_spec)

    args.output.write_text(json.dumps(output_spec, indent=2) + "\n")
    component_counts = ", ".join(
        f"{len(names)} {section}" for section, names in sorted(output_spec["components"].items())
    )
    print(f"Wrote {component_counts} and {len(output_spec['paths'])} paths to {args.output}")


if __name__ == "__main__":
    main()
