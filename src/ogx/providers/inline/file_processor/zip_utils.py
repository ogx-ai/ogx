# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import zipfile

from fastapi import HTTPException

MAX_ZIP_DECOMPRESSED_BYTES = 25 * 1024 * 1024  # 25MiB
MAX_ZIP_ENTRIES = 1000


def validate_zip_content(content: bytes, filename: str) -> None:
    """Reject ZIP archives that exceed decompression limits.

    Call before processing any file that may be ZIP-based (DOCX, PPTX, XLSX, etc.).
    """
    if not zipfile.is_zipfile(io.BytesIO(content)):
        return

    with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
        entries = zf.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Failed to process file '{filename}': ZIP contains {len(entries)} entries, "
                    f"exceeding the limit of {MAX_ZIP_ENTRIES}"
                ),
            )
        total_size = sum(e.file_size for e in entries)
        if total_size > MAX_ZIP_DECOMPRESSED_BYTES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Failed to process file '{filename}': ZIP decompressed size "
                    f"({total_size} bytes) exceeds the limit of {MAX_ZIP_DECOMPRESSED_BYTES} bytes"
                ),
            )
