# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from contextvars import ContextVar

from ogx.core.request_headers import PROVIDER_DATA_VAR

TEST_CONTEXT: ContextVar[str | None] = ContextVar("ogx_test_context", default=None)


def get_test_context() -> str | None:
    """Get the current test context identifier.

    Returns:
        The test context string, or None if not set.
    """
    return TEST_CONTEXT.get()


def set_test_context(value: str | None):
    """Set the test context identifier for the current async context.

    Args:
        value: The test context string to set, or None to clear.

    Returns:
        A token that can be used to reset the context variable.
    """
    return TEST_CONTEXT.set(value)


def reset_test_context(token) -> None:
    """Reset the test context to its previous value using a token from set_test_context.

    Args:
        token: The token returned by a previous set_test_context call.
    """
    TEST_CONTEXT.reset(token)


def sync_test_context_from_provider_data():
    """Sync test context from provider data when running in server test mode."""
    if "OGX_TEST_INFERENCE_MODE" not in os.environ:
        return None

    stack_config_type = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "library_client")
    if stack_config_type != "server":
        return None

    try:
        provider_data = PROVIDER_DATA_VAR.get()
    except LookupError:
        provider_data = None

    if provider_data and "__test_id" in provider_data:
        return TEST_CONTEXT.set(provider_data["__test_id"])

    return None


def is_debug_mode() -> bool:
    """Check if test recording debug mode is enabled via OGX_TEST_DEBUG env var."""
    return os.environ.get("OGX_TEST_DEBUG", "").lower() in ("1", "true", "yes")
