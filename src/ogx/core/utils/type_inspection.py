# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Utility functions for type inspection and parameter handling.
"""

import inspect
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def is_unwrapped_body_param(param_type: Any) -> bool:
    """
    Check if a parameter type represents an unwrapped body parameter.
    An unwrapped body parameter is an Annotated type with Body(embed=False)

    This is used to determine whether request parameters should be flattened
    in OpenAPI specs and client libraries (matching FastAPI's embed=False behavior).

    Args:
        param_type: The parameter type annotation to check

    Returns:
        True if the parameter should be treated as an unwrapped body parameter
    """
    # Check if it's Annotated with Body(embed=False)
    if get_origin(param_type) is typing.Annotated:
        args = get_args(param_type)
        base_type = args[0]
        metadata = args[1:]

        # Look for Body annotation; treat embed=None (default) as unwrapped
        # Body() returns a FieldInfo object, so we check for that type and the embed attribute
        for item in metadata:
            if isinstance(item, FieldInfo) and hasattr(item, "embed") and item.embed is not True:
                return inspect.isclass(base_type) and issubclass(base_type, BaseModel)

    return False


def is_body_param(param_type: Any) -> bool:
    """
    Check if a parameter type represents a body parameter (Annotated with Body()).
    """
    if get_origin(param_type) is typing.Annotated:
        args = get_args(param_type)
        base_type = args[0]
        metadata = args[1:]
        for item in metadata:
            if isinstance(item, FieldInfo):
                return inspect.isclass(base_type) and issubclass(base_type, BaseModel)
    return False
