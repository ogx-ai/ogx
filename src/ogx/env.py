# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os


class MissingCredentialError(Exception):
    """Raised when a required credential is not found in the environment."""

    pass


def get_env_or_fail(key: str) -> str:
    """Get environment variable or raise helpful error"""
    value = os.getenv(key)
    if not value:
        raise MissingCredentialError(
            f"\nMissing {key} in environment. Please set it using one of these methods:"
            f"\n1. Export in shell: export {key}=your-key"
            f"\n2. Create .env file in project root with: {key}=your-key"
            f"\n3. Pass directly to pytest: pytest --env {key}=your-key"
        )
    return value
