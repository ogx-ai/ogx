# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared exception mappings for HTTP status code translation.

This module provides a single source of truth for mapping exception types
to HTTP status codes. It is used by:
- server.py: to translate exceptions to HTTPException responses
- testing/exception_utils.py: to reconstruct exceptions during test replay
"""

import asyncio

import httpx
from fastapi import HTTPException
from openai import BadRequestError

from ogx.core.access_control.access_control import AccessDeniedError
from ogx.core.datatypes import AuthenticationRequiredError

# Maps exception type -> (status_code, fallback_detail)
# The exception's own message is used when present. The fallback is only
# used when the exception has no message.
EXCEPTION_MAP: dict[type, tuple[int, str]] = {
    ValueError: (httpx.codes.BAD_REQUEST, "Invalid value"),
    BadRequestError: (httpx.codes.BAD_REQUEST, "Bad request"),
    PermissionError: (httpx.codes.FORBIDDEN, "Permission denied"),
    AccessDeniedError: (httpx.codes.FORBIDDEN, "Permission denied"),
    ConnectionError: (httpx.codes.BAD_GATEWAY, "Connection error"),
    httpx.ConnectError: (httpx.codes.BAD_GATEWAY, "Connection error"),
    TimeoutError: (httpx.codes.GATEWAY_TIMEOUT, "Operation timed out"),
    asyncio.TimeoutError: (httpx.codes.GATEWAY_TIMEOUT, "Operation timed out"),
    NotImplementedError: (httpx.codes.NOT_IMPLEMENTED, "Not implemented"),
    AuthenticationRequiredError: (httpx.codes.UNAUTHORIZED, "Authentication required"),
}

# For deserialization by class name (used by testing/exception_utils.py)
EXCEPTION_TYPES_BY_NAME: dict[str, type[Exception]] = {cls.__name__: cls for cls in EXCEPTION_MAP}


def translate_exception_to_http(exc: Exception) -> HTTPException | None:
    """Translate an exception to an HTTPException using the mapping.

    Walks up the exception's inheritance chain (MRO) and checks for a match
    in the mapping. This is O(k) where k is the inheritance depth, with O(1)
    dict lookup at each level.

    Returns None if the exception type is not in the mapping.
    """
    for cls in type(exc).__mro__:
        if cls in EXCEPTION_MAP:
            status_code, fallback = EXCEPTION_MAP[cls]
            detail = str(exc) or fallback
            return HTTPException(status_code=status_code, detail=detail)
    return None
