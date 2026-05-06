# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re

_SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")


def sanitize_content_disposition_filename(filename: str) -> str:
    """Sanitize *filename* for safe storage and inclusion in HTTP headers.

    Applied at **upload time** so the clean value is persisted in the database
    and returned in API responses, as well as at **download time** in the
    ``Content-Disposition`` header.

    The function strips null bytes and path separators, collapses ``..``
    sequences, prevents hidden-file names (leading dot), and replaces any
    remaining characters outside an ASCII alphanumeric allowlist.  Returns
    ``"download"`` when the result would otherwise be empty.
    """
    filename = filename.replace("\x00", "")
    filename = filename.replace('"', "_")  # prevent Content-Disposition header injection
    filename = filename.replace("/", "_").replace("\\", "_")
    filename = filename.replace("..", "_")
    if filename.startswith("."):
        filename = "_" + filename[1:]
    filename = _SAFE_FILENAME_PATTERN.sub("_", filename)
    return filename or "download"
