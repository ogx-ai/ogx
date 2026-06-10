# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.inline.skills.builtin.manifest import parse_skill_manifest


class TestParseSkillManifest:
    def test_valid_manifest(self):
        content = """---
name: my-skill
description: A test skill
version: "1.0"
---
Use this skill to do things.
"""
        manifest = parse_skill_manifest(content)
        assert manifest.name == "my-skill"
        assert manifest.description == "A test skill"
        assert manifest.version == "1.0"
        assert manifest.instructions == "Use this skill to do things."

    def test_manifest_with_tools(self):
        content = """---
name: analyzer
tools:
  - name: run_analysis
    type: shell
---
Instructions here.
"""
        manifest = parse_skill_manifest(content)
        assert manifest.name == "analyzer"
        assert manifest.tools is not None
        assert len(manifest.tools) == 1
        assert manifest.tools[0]["name"] == "run_analysis"

    def test_no_frontmatter(self):
        content = "Just plain instructions without frontmatter."
        manifest = parse_skill_manifest(content)
        assert manifest.name is None
        assert manifest.instructions == content

    def test_empty_content(self):
        manifest = parse_skill_manifest("")
        assert manifest.name is None
        assert manifest.instructions == ""

    def test_frontmatter_only(self):
        content = """---
name: minimal
description: No instructions
---"""
        manifest = parse_skill_manifest(content)
        assert manifest.name == "minimal"
        assert manifest.description == "No instructions"
        assert manifest.instructions == ""

    def test_missing_closing_fence(self):
        content = """---
name: broken
description: Missing closing fence
"""
        manifest = parse_skill_manifest(content)
        assert manifest.name is None
        assert manifest.instructions == content.strip()

    def test_empty_frontmatter(self):
        content = """---
---
Instructions only."""
        manifest = parse_skill_manifest(content)
        assert manifest.name is None
        assert manifest.instructions == "Instructions only."

    def test_multiline_instructions(self):
        content = """---
name: multi
---
Line one.

Line two.

Line three."""
        manifest = parse_skill_manifest(content)
        assert manifest.name == "multi"
        assert "Line one." in manifest.instructions
        assert "Line three." in manifest.instructions
