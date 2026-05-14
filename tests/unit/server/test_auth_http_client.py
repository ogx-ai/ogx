# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx

from ogx.core.datatypes import (
    CustomAuthConfig,
)
from ogx.core.server.auth_providers import (
    CustomAuthProvider,
)


class TestCustomAuthProviderClient:
    def test_client_created_with_timeout(self):
        config = CustomAuthConfig(endpoint="http://auth.example.com/validate")
        provider = CustomAuthProvider(config)
        assert isinstance(provider._client, httpx.AsyncClient)
        assert provider._client.timeout.connect == 5.0
        assert provider._client.timeout.read == 10.0
        assert provider._client.timeout.write == 10.0
        assert provider._client.timeout.pool == 10.0

    def test_client_reused_across_provider_lifetime(self):
        config = CustomAuthConfig(endpoint="http://auth.example.com/validate")
        provider = CustomAuthProvider(config)
        client_ref = provider._client
        assert provider._client is client_ref

    async def test_close_shuts_down_client(self):
        config = CustomAuthConfig(endpoint="http://auth.example.com/validate")
        provider = CustomAuthProvider(config)
        assert not provider._client.is_closed
        await provider.close()
        assert provider._client.is_closed
