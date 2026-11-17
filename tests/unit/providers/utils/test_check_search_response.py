# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx2
import pytest

from ogx.providers.utils.common.web_search import check_search_response


def _response(status_code: int, url: str = "https://example.com/search") -> httpx2.Response:
    return httpx2.Response(status_code, request=httpx2.Request("GET", url))


def test_ok_response_passes():
    check_search_response(_response(200), "Test Search")


def test_401_raises_key_error():
    with pytest.raises(RuntimeError, match="the API key is missing or invalid"):
        check_search_response(_response(401), "Test Search")


def test_403_raises_generic_entitlement_error():
    with pytest.raises(RuntimeError, match="lacks the required entitlement"):
        check_search_response(_response(403), "Test Search")


def test_403_forbidden_detail_overrides_generic_message():
    with pytest.raises(RuntimeError, match="vendor detail"):
        check_search_response(_response(403), "Test Search", forbidden_detail="vendor detail")


def test_other_errors_use_raise_for_status():
    with pytest.raises(httpx2.HTTPStatusError):
        check_search_response(_response(500), "Test Search")
