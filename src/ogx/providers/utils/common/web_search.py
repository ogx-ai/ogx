# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx2


def check_search_response(response: httpx2.Response, provider_name: str, forbidden_detail: str | None = None) -> None:
    """Raise an actionable error for auth-class search failures.

    401 and 403 are configuration-class failures (a missing or invalid API
    key, or an account that lacks the entitlement for the requested
    capability) rather than transient faults. Since ``WebSearchResult``
    carries no error fields, backends report them by raising; the
    ``RuntimeError`` message is what the executor forwards to the model, so
    it states the reason and the fix instead of a raw status line.

    ``forbidden_detail`` lets a backend replace the generic 403 explanation
    with a vendor-specific one (e.g. Nimble's search_depth tier gating).
    Other non-2xx responses are left to ``raise_for_status()``.
    """
    if response.status_code == 401:
        raise RuntimeError(f"Failed to query {provider_name}: the API key is missing or invalid.")
    if response.status_code == 403:
        detail = forbidden_detail or "the API key or account lacks the required entitlement for this request"
        raise RuntimeError(f"Failed to query {provider_name}: {detail}.")
    response.raise_for_status()
