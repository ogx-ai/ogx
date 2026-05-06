# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import LlamaStackClient

from ogx.core.library_client import OGXAsLibraryClient


class TestProviders:
    def test_providers(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        provider_list = ogx_client.providers.list()
        assert provider_list is not None
        assert len(provider_list) > 0

        for provider in provider_list:
            pid = provider.provider_id
            provider = ogx_client.providers.retrieve(pid)
            assert provider is not None
