# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def auth_error_message(base_url: str, status_code: int) -> str:
    """Error text for a server that rejected the credentials (HTTP 401/403)."""
    return (
        f"Failed to authenticate with the OGX server at {base_url} (HTTP {status_code}).\n"
        "Pass the server's API key with 'ogx connect --api-key <key>' or set OGX_API_KEY."
    )
