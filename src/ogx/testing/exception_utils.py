# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared exception handling utilities for recording/replaying exceptions.

This module provides utilities for serializing and deserializing exceptions
during API recording/replay. The exception types handled here use the shared
mapping from mapping.py, ensuring consistency between runtime
exception handling and test replay.
"""

from typing import Any, Protocol, TypeGuard

import httpx

from ogx.core.exceptions.mapping import EXCEPTION_TYPES_BY_NAME
from ogx.testing.providers import GenericProviderError, create_provider_error, detect_provider
from ogx_api.common.errors import OGXError

__all__ = [
    "GenericProviderError",
    "GenericOGXError",
    "ProviderSDKException",
    "deserialize_exception",
    "is_provider_sdk_exception",
    "serialize_exception",
]


class ProviderSDKException(Protocol):
    """Protocol for provider SDK exceptions with status_code attribute."""

    status_code: int
    body: dict | None


class GenericOGXError(OGXError):
    """A generic OGXError for replay when exact type can't be reconstructed."""

    def __init__(self, status_code_value: int, message: str = ""):
        super().__init__(message)
        # Override the class variable with an instance attribute
        self.status_code = httpx.codes(status_code_value)


def is_provider_sdk_exception(exc: Exception) -> TypeGuard[ProviderSDKException]:
    """Check if exception is a provider SDK exception (e.g., OpenAI APIStatusError).

    Provider SDK exceptions have a status_code attribute that indicates the HTTP
    status code from the upstream provider. This matches the duck-typing used
    in server.translate_exception().
    """
    return hasattr(exc, "status_code") and isinstance(getattr(exc, "status_code", None), int)


def serialize_exception(exc: Exception) -> dict[str, Any]:
    """Serialize an exception for recording.

    Categories:
    - ogx: OGXError subclasses (internal errors)
    - provider_sdk: Exceptions with status_code attr (OpenAI, etc.)
    - builtin: Python built-in exceptions handled by translate_exception
    - unknown: Everything else (will replay as generic Exception)
    """
    exc_type = type(exc).__name__
    message = str(exc)

    # Check categories in order of specificity
    if isinstance(exc, OGXError):
        return {
            "category": "ogx",
            "type": exc_type,
            "message": message,
            "status_code": int(exc.status_code),
        }
    elif is_provider_sdk_exception(exc):
        error_message = getattr(exc, "error", message)
        return {
            "category": "provider_sdk",
            "provider": detect_provider(exc),
            "type": exc_type,
            "message": error_message,
            "status_code": exc.status_code,
            "body": getattr(exc, "body", None),
        }
    elif exc_type in EXCEPTION_TYPES_BY_NAME:
        return {
            "category": "builtin",
            "type": exc_type,
            "message": message,
        }
    else:
        return {
            "category": "unknown",
            "type": exc_type,
            "message": message,
        }


def deserialize_exception(data: dict[str, Any]) -> Exception:
    """Reconstruct an exception from recorded data.

    The reconstructed exception will have the same interface that
    server.translate_exception() expects, ensuring consistent behavior
    between live and replay modes.
    """
    category = data.get("category", "unknown")
    exc_type = data.get("type", "Exception")
    message = data.get("message", "Unknown error")

    if category == "ogx":
        status_code = data.get("status_code", 500)
        return GenericOGXError(status_code, message)

    elif category == "provider_sdk":
        provider = data.get("provider", "unknown")
        return create_provider_error(
            provider=provider,
            status_code=data.get("status_code", 500),
            body=data.get("body"),
            message=message,
        )

    elif category == "builtin":
        if exc_type in EXCEPTION_TYPES_BY_NAME:
            return EXCEPTION_TYPES_BY_NAME[exc_type](message)
        return Exception(message)

    else:
        # Unknown category - return generic exception
        return Exception(message)
