# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import zipfile
from io import BytesIO
from pathlib import PurePosixPath

from ogx_api.skills.models import (
    MAX_FILES_PER_VERSION,
    MAX_UNCOMPRESSED_FILE_SIZE_BYTES,
    MAX_ZIP_SIZE_BYTES,
    SkillManifest,
)

from .manifest import parse_skill_manifest

_SKILL_MD = "SKILL.md"


def _has_path_traversal(filename: str) -> bool:
    """Check if a zip entry filename attempts path traversal."""
    if filename.startswith("/"):
        return True
    return ".." in PurePosixPath(filename).parts


def validate_skill_zip(content: bytes) -> tuple[SkillManifest, list[str]]:
    """Validate a skill zip bundle and extract its manifest.

    Returns:
        Tuple of (parsed manifest, list of file paths in the archive).

    Raises:
        ValueError: If the bundle fails any validation check.
    """
    if len(content) > MAX_ZIP_SIZE_BYTES:
        raise ValueError(
            f"Failed to validate skill bundle: zip size {len(content)} bytes "
            f"exceeds maximum of {MAX_ZIP_SIZE_BYTES} bytes"
        )

    try:
        zf = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile as e:
        raise ValueError("Failed to validate skill bundle: file is not a valid zip archive") from e

    with zf:
        entries = zf.infolist()

        if len(entries) > MAX_FILES_PER_VERSION:
            raise ValueError(
                f"Failed to validate skill bundle: archive contains {len(entries)} files, "
                f"maximum is {MAX_FILES_PER_VERSION}"
            )

        file_paths: list[str] = []
        skill_md_content: str | None = None

        for entry in entries:
            if _has_path_traversal(entry.filename):
                raise ValueError(f"Failed to validate skill bundle: path traversal detected in '{entry.filename}'")

            if entry.file_size > MAX_UNCOMPRESSED_FILE_SIZE_BYTES:
                raise ValueError(
                    f"Failed to validate skill bundle: '{entry.filename}' uncompressed size "
                    f"{entry.file_size} bytes exceeds maximum of {MAX_UNCOMPRESSED_FILE_SIZE_BYTES} bytes"
                )

            if not entry.is_dir():
                file_paths.append(entry.filename)

            if entry.filename == _SKILL_MD:
                skill_md_content = zf.read(entry.filename).decode("utf-8")

    if skill_md_content is None:
        raise ValueError("Failed to validate skill bundle: SKILL.md not found at archive root")

    manifest = parse_skill_manifest(skill_md_content)
    if not manifest.name:
        raise ValueError("Failed to validate skill bundle: SKILL.md frontmatter must include 'name'")

    return manifest, file_paths
