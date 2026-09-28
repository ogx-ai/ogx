# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""stamp_test_id_into_headers() is the single implementation shared by _inject_test_id
(api_recorder.py, real HTTP requests) and the in-process library-client call paths
(library_client.py), so the header parse/merge logic can't drift between the two (#6668).
It must tolerate the same malformed provider-data header parse_request_provider_data()
already defends against, since that's what a downstream request_provider_data_context()
call does with this same header anyway.
"""

import json

import httpx
import pytest

from ogx.core.request_headers import stamp_test_id_into_headers
from ogx.core.testing_context import reset_test_context, set_test_context

TEST_ID = "tests/unit/core/test_request_headers.py::test_it"


@pytest.fixture
def active_test_context():
    token = set_test_context(TEST_ID)
    yield TEST_ID
    reset_test_context(token)


class TestStampTestIdIntoHeaders:
    def test_no_op_without_an_active_test_context(self):
        headers: dict[str, str] = {}

        stamp_test_id_into_headers(headers)

        assert headers == {}

    def test_adds_the_header_when_absent(self, active_test_context):
        headers: dict[str, str] = {}

        stamp_test_id_into_headers(headers)

        assert json.loads(headers["X-OGX-Provider-Data"]) == {"__test_id": active_test_context}

    def test_merges_into_valid_existing_json(self, active_test_context):
        headers = {"X-OGX-Provider-Data": json.dumps({"other_key": "value"})}

        stamp_test_id_into_headers(headers)

        assert json.loads(headers["X-OGX-Provider-Data"]) == {
            "other_key": "value",
            "__test_id": active_test_context,
        }

    def test_invalid_json_is_discarded_rather_than_raising(self, active_test_context):
        headers = {"X-OGX-Provider-Data": "not-valid-json"}

        stamp_test_id_into_headers(headers)

        assert json.loads(headers["X-OGX-Provider-Data"]) == {"__test_id": active_test_context}

    def test_non_dict_json_is_discarded_rather_than_raising(self, active_test_context):
        headers = {"X-OGX-Provider-Data": json.dumps(["a", "list", "not", "an", "object"])}

        stamp_test_id_into_headers(headers)

        assert json.loads(headers["X-OGX-Provider-Data"]) == {"__test_id": active_test_context}

    def test_lowercase_header_key_is_recognized_and_preserved(self, active_test_context):
        headers = {"x-ogx-provider-data": json.dumps({"other_key": "value"})}

        stamp_test_id_into_headers(headers)

        assert "X-OGX-Provider-Data" not in headers
        assert json.loads(headers["x-ogx-provider-data"]) == {
            "other_key": "value",
            "__test_id": active_test_context,
        }

    def test_works_on_httpx_headers_too(self, active_test_context):
        """_inject_test_id (api_recorder.py) calls this with an httpx.Request's case-insensitive
        Headers object rather than a plain dict; it must support both mapping types."""
        headers = httpx.Headers({"x-ogx-provider-data": json.dumps({"other_key": "value"})})

        stamp_test_id_into_headers(headers)

        assert json.loads(headers["X-OGX-Provider-Data"]) == {
            "other_key": "value",
            "__test_id": active_test_context,
        }
