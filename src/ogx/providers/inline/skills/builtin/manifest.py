# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import yaml

from ogx_api.skills.models import SkillManifest

_FENCE = "---"


def parse_skill_manifest(content: str) -> SkillManifest:
    """Parse a SKILL.md file into a SkillManifest.

    Expected format:
        ---
        name: my-skill
        description: Does something useful
        ---
        Instructions for the model go here.
    """
    stripped = content.strip()
    if not stripped.startswith(_FENCE):
        return SkillManifest(instructions=stripped)

    after_first_fence = stripped[len(_FENCE) :]
    end_idx = after_first_fence.find(_FENCE)
    if end_idx == -1:
        return SkillManifest(instructions=stripped)

    frontmatter_text = after_first_fence[:end_idx]
    instructions = after_first_fence[end_idx + len(_FENCE) :].strip()

    frontmatter = yaml.safe_load(frontmatter_text)
    if not isinstance(frontmatter, dict):
        return SkillManifest(instructions=instructions)

    return SkillManifest(
        name=frontmatter.get("name"),
        description=frontmatter.get("description"),
        version=frontmatter.get("version"),
        tools=frontmatter.get("tools"),
        instructions=instructions,
    )
