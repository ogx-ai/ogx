# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.inline.responses.builtin.responses.openai_responses import OpenAIResponsesImpl


def _make_skill_zip(name: str, description: str, instructions: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = f"---\nname: {name}\ndescription: {description}\n---\n{instructions}"
        zf.writestr("SKILL.md", manifest)
    return buf.getvalue()


def _make_mock_skills_api(skill_name: str, skill_description: str, instructions: str) -> AsyncMock:
    api = AsyncMock()
    skill = MagicMock()
    skill.name = skill_name
    skill.description = skill_description
    skill.default_version = "1"
    api.get_skill = AsyncMock(return_value=skill)

    zip_bytes = _make_skill_zip(skill_name, skill_description, instructions)
    resp = MagicMock()
    resp.body = zip_bytes
    api.get_skill_version_content = AsyncMock(return_value=resp)
    return api


class TestResolveSkillInstructions:
    async def test_skill_instructions_include_manifest_body(self):
        skills_api = _make_mock_skills_api(
            "code-reviewer",
            "Reviews code for security issues",
            "Always check for SQL injection, XSS, and auth bypass.",
        )
        impl = OpenAIResponsesImpl.__new__(OpenAIResponsesImpl)
        impl.skills_api = skills_api

        messages = await impl._resolve_skill_instructions(["skill-123"])

        assert len(messages) == 1
        content = messages[0].content
        assert "## Skill: code-reviewer" in content
        assert "Reviews code for security issues" in content
        assert "### Instructions" in content
        assert "Always check for SQL injection, XSS, and auth bypass." in content

    async def test_skill_instructions_multiple_skills(self):
        api1 = _make_mock_skills_api("analyzer", "Analyzes data", "Run analyze.py")
        api2 = _make_mock_skills_api("reviewer", "Reviews code", "Check for bugs")

        call_count = 0

        async def mock_get_skill(skill_id):
            nonlocal call_count
            call_count += 1
            if "analyzer" in skill_id:
                return api1.get_skill.return_value
            return api2.get_skill.return_value

        async def mock_get_content(skill_id, version):
            if "analyzer" in skill_id:
                return api1.get_skill_version_content.return_value
            return api2.get_skill_version_content.return_value

        combined_api = AsyncMock()
        combined_api.get_skill = mock_get_skill
        combined_api.get_skill_version_content = mock_get_content

        impl = OpenAIResponsesImpl.__new__(OpenAIResponsesImpl)
        impl.skills_api = combined_api

        messages = await impl._resolve_skill_instructions(["skill-analyzer", "skill-reviewer"])

        assert len(messages) == 1
        content = messages[0].content
        assert "analyzer" in content.lower()
        assert "reviewer" in content.lower()

    async def test_skill_instructions_empty_list(self):
        impl = OpenAIResponsesImpl.__new__(OpenAIResponsesImpl)
        impl.skills_api = AsyncMock()

        messages = await impl._resolve_skill_instructions([])
        assert messages == []

    async def test_skill_instructions_invalid_id_raises(self):
        api = AsyncMock()
        api.get_skill = AsyncMock(side_effect=ValueError("Failed to find skill"))

        impl = OpenAIResponsesImpl.__new__(OpenAIResponsesImpl)
        impl.skills_api = api

        with pytest.raises(ValueError, match="Failed to find skill"):
            await impl._resolve_skill_instructions(["nonexistent-skill"])
