# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.client import create_api_client_class
from ogx_api.schema_utils import webmethod


class _DummyProtocol:
    @webmethod(route="/echo", method="GET", level="v1")
    async def echo(self, query: str) -> dict[str, str]:
        raise NotImplementedError


def test_request_params_do_not_override_client_timeout() -> None:
    client_class = create_api_client_class(_DummyProtocol)
    client = client_class("https://example.com")
    params = client.httpx_request_params("echo", query="hello")
    assert "timeout" not in params
