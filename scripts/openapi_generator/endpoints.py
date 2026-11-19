# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Endpoint generation logic for FastAPI OpenAPI generation.
"""

import re
from typing import Any


def _extract_path_parameters(path: str) -> list[dict[str, Any]]:
    """Extract path parameters from a URL path and return them as OpenAPI parameter definitions."""
    matches = re.findall(r"\{([^}:]+)(?::[^}]+)?\}", path)
    return [
        {
            "name": param_name,
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
            "description": f"Path parameter: {param_name}",
        }
        for param_name in matches
    ]
