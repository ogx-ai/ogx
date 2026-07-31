# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAPI generator module for OGX.

This module provides functionality to generate OpenAPI specifications
from FastAPI applications.
"""

__all__ = ["generate_openapi_spec", "main"]


def __getattr__(name: str):
    if name in {"generate_openapi_spec", "main"}:
        from .main import generate_openapi_spec as _gos
        from .main import main as _main

        return {"generate_openapi_spec": _gos, "main": _main}[name]
    raise AttributeError(name)
