# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from fastapi import Request
from pydantic import BaseModel, ValidationError

from ogx_api import ExpiresAfter


async def parse_pydantic_from_form[T: BaseModel](request: Request, field_name: str, model_class: type[T]) -> T | None:
    """
    Generic parser to extract a Pydantic model from multipart form data.
    Handles both bracket notation (field[attr1], field[attr2]) and JSON string format.

    Args:
        request: The FastAPI request object
        field_name: The name of the field in the form data (e.g., "expires_after")
        model_class: The Pydantic model class to parse into

    Returns:
        An instance of model_class if parsing succeeds, None otherwise

    Example:
        expires_after = await parse_pydantic_from_form(
            request, "expires_after", ExpiresAfter
        )
    """
    form = await request.form()

    # Check for bracket notation first (e.g., expires_after[anchor], expires_after[seconds])
    bracket_data = {}
    prefix = f"{field_name}["
    for key in form.keys():
        if key.startswith(prefix) and key.endswith("]"):
            # Extract the attribute name from field_name[attr]
            attr = key[len(prefix) : -1]
            bracket_data[attr] = form[key]

    if bracket_data:
        try:
            return model_class(**bracket_data)
        except (ValidationError, TypeError):
            pass

    # Check for JSON string format
    if field_name in form:
        value = form[field_name]
        if isinstance(value, str):
            try:
                data = json.loads(value)
                return model_class(**data)
            except (json.JSONDecodeError, TypeError, ValidationError):
                pass

    return None


async def parse_expires_after(request: Request) -> ExpiresAfter | None:
    """
    Dependency to parse expires_after from multipart form data.
    Handles both bracket notation (expires_after[anchor], expires_after[seconds])
    and JSON string format.
    """
    return await parse_pydantic_from_form(request, "expires_after", ExpiresAfter)
