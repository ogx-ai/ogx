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
    lines = content.split("\n")

    if not lines or lines[0].strip() != _FENCE:
        return SkillManifest(instructions=content.strip())

    closing_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FENCE:
            closing_idx = i
            break

    if closing_idx is None:
        return SkillManifest(instructions=content.strip())

    frontmatter_text = "\n".join(lines[1:closing_idx])
    instructions = "\n".join(lines[closing_idx + 1 :]).strip()

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
