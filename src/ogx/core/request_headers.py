# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import contextvars
import json
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

from starlette.types import Scope

from ogx.core.datatypes import User
from ogx.log import get_logger

from .utils.dynamic import instantiate_class_type

if TYPE_CHECKING:
    from ogx_api import ProviderSpec

log = get_logger(name=__name__, category="core")

# Context variable for request provider data and auth attributes
PROVIDER_DATA_VAR: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("provider_data", default=None)


class RequestProviderDataContext(AbstractContextManager[None]):
    """Context manager for request provider data"""

    def __init__(self, provider_data: dict[str, Any] | None = None, user: User | None = None) -> None:
        if provider_data is not None and not isinstance(provider_data, dict):
            log.error("Provider data must be a JSON object")
            provider_data = None
        self.provider_data = provider_data or {}
        if user:
            self.provider_data["__authenticated_user"] = user

        self.token: contextvars.Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> None:
        # Save the current value and set the new one
        self.token = PROVIDER_DATA_VAR.set(self.provider_data)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore the previous value
        if self.token is not None:
            PROVIDER_DATA_VAR.reset(self.token)


class NeedsRequestProviderData:
    """Mixin for providers that require per-request provider data from request headers."""

    __provider_spec__: "ProviderSpec"

    def get_request_provider_data(self) -> Any:
        spec = self.__provider_spec__  # type: ignore[attr-defined]
        if not spec:
            raise ValueError(f"Provider spec not set on {self.__class__}")

        provider_type = spec.provider_type
        validator_class = spec.provider_data_validator
        if not validator_class:
            raise ValueError(f"Provider {provider_type} does not have a validator")

        val = PROVIDER_DATA_VAR.get()
        if not val:
            return None

        validator = instantiate_class_type(validator_class)  # type: ignore[no-untyped-call]
        try:
            provider_data = validator(**val)
            return provider_data
        except Exception as e:
            log.error(f"Error parsing provider data: {e}")
            return None


def parse_request_provider_data(headers: dict[str, str]) -> dict[str, Any] | None:
    """Parse provider data from request headers"""
    keys = [
        "X-OGX-Provider-Data",
        "x-ogx-provider-data",
    ]
    val = None
    for key in keys:
        val = headers.get(key, None)
        if val:
            break

    if not val:
        return None

    try:
        parsed = json.loads(val)
    except json.JSONDecodeError:
        log.error("Provider data not encoded as a JSON object!")
        return None

    if parsed is None:
        return None

    if not isinstance(parsed, dict):
        log.error("Provider data must be encoded as a JSON object")
        return None

    return cast(dict[str, Any], parsed)


def request_provider_data_context(headers: dict[str, str], user: User | None = None) -> AbstractContextManager[None]:
    """Context manager that sets request provider data from headers and user for the duration of the context"""
    provider_data = parse_request_provider_data(headers)
    return RequestProviderDataContext(provider_data, user)


def get_authenticated_user() -> User | None:
    """Helper to retrieve auth attributes from the provider data context"""
    provider_data = PROVIDER_DATA_VAR.get()
    if not provider_data:
        return None
    return provider_data.get("__authenticated_user")


def user_from_scope(scope: Scope) -> User | None:
    """Create a User object from ASGI scope data (set by authentication middleware)"""
    user_attributes = scope.get("user_attributes", {})
    principal = scope.get("principal", "")

    # auth not enabled
    if not principal and not user_attributes:
        return None

    return User(principal=principal, attributes=user_attributes)
