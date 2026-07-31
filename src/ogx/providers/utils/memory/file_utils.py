# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os

from ogx_api import URL


def data_url_from_file(file_path: str) -> URL:
    """Create a data URL from a local file path.

    Args:
        file_path: path to the file on disk

    Returns:
        A URL object containing the base64-encoded data URL
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return URL(uri=data_url)
