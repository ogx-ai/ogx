# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import zipfile

import pytest

from ogx.providers.inline.skills.builtin.validation import validate_skill_zip
from ogx_api.skills.models import MAX_FILES_PER_VERSION, MAX_UNCOMPRESSED_FILE_SIZE_BYTES


def _make_zip(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()


VALID_SKILL_MD = """---
name: test-skill
description: A test skill
---
Do the thing."""


class TestValidateSkillZip:
    def test_valid_zip(self):
        content = _make_zip({"SKILL.md": VALID_SKILL_MD, "run.py": "print('hello')"})
        manifest, file_paths = validate_skill_zip(content)
        assert manifest.name == "test-skill"
        assert manifest.description == "A test skill"
        assert "SKILL.md" in file_paths
        assert "run.py" in file_paths

    def test_missing_skill_md(self):
        content = _make_zip({"README.md": "No skill manifest here."})
        with pytest.raises(ValueError, match="SKILL.md not found"):
            validate_skill_zip(content)

    def test_skill_md_missing_name(self):
        content = _make_zip({"SKILL.md": "---\ndescription: no name\n---\nInstructions."})
        with pytest.raises(ValueError, match="must include 'name'"):
            validate_skill_zip(content)

    def test_not_a_zip(self):
        with pytest.raises(ValueError, match="not a valid zip archive"):
            validate_skill_zip(b"this is not a zip file")

    def test_path_traversal_dotdot(self):
        content = _make_zip({"SKILL.md": VALID_SKILL_MD, "../etc/passwd": "root:x:0"})
        with pytest.raises(ValueError, match="path traversal"):
            validate_skill_zip(content)

    def test_path_traversal_absolute(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SKILL.md", VALID_SKILL_MD)
            zf.writestr("/etc/shadow", "bad")
        content = buf.getvalue()
        with pytest.raises(ValueError, match="path traversal"):
            validate_skill_zip(content)

    def test_too_many_files(self):
        files = {"SKILL.md": VALID_SKILL_MD}
        for i in range(MAX_FILES_PER_VERSION + 1):
            files[f"file_{i}.txt"] = "x"
        content = _make_zip(files)
        with pytest.raises(ValueError, match="maximum is"):
            validate_skill_zip(content)

    def test_file_too_large(self):
        files = {
            "SKILL.md": VALID_SKILL_MD,
            "big.bin": b"\x00" * (MAX_UNCOMPRESSED_FILE_SIZE_BYTES + 1),
        }
        content = _make_zip(files)
        with pytest.raises(ValueError, match="uncompressed size"):
            validate_skill_zip(content)

    def test_zip_size_limit(self):
        from ogx_api.skills.models import MAX_ZIP_SIZE_BYTES

        oversized = b"\x00" * (MAX_ZIP_SIZE_BYTES + 1)
        with pytest.raises(ValueError, match="zip size"):
            validate_skill_zip(oversized)
