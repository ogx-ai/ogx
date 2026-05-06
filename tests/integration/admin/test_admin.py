# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import LlamaStackClient

from ogx.core.library_client import OGXAsLibraryClient


class TestAdmin:
    def test_admin_providers_list(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        provider_list = ogx_client.alpha.admin.list_providers()
        assert provider_list is not None
        assert len(provider_list) > 0

        for provider in provider_list:
            pid = provider.provider_id
            provider = ogx_client.alpha.admin.inspect_provider(pid)
            assert provider is not None

    def test_health(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        health = ogx_client.alpha.admin.health()
        assert health is not None
        assert health.status == "OK"

    def test_version(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        version = ogx_client.alpha.admin.version()
        assert version is not None
        assert version.version is not None

    def test_list_routes_default(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        """Test list_routes with default filter (non-deprecated v1 routes)."""
        routes = ogx_client.alpha.admin.list_routes()
        assert routes is not None
        assert len(routes) > 0

        # All routes should be non-deprecated
        # Check that we don't see any /openai/ routes (which are deprecated)
        openai_routes = [r for r in routes if "/openai/" in r.route]
        assert len(openai_routes) == 0, "Default filter should not include deprecated /openai/ routes"

        # Should see standard v1 routes like /inspect/routes, /health, /version
        paths = [r.route for r in routes]
        assert "/inspect/routes" in paths or "/v1/inspect/routes" in paths
        assert "/health" in paths or "/v1/health" in paths

    def test_list_routes_filter_by_deprecated(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        """Test list_routes with deprecated filter."""
        routes = ogx_client.alpha.admin.list_routes(api_filter="deprecated")
        assert routes is not None

        # When filtering for deprecated, we should get deprecated routes
        # Verify we get some deprecated routes (e.g., /toolgroups, /shields, /models, etc.)
        assert len(routes) > 0, "Deprecated filter should return some deprecated routes"

    def test_list_routes_filter_by_v1(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        """Test list_routes with v1 filter."""
        routes = ogx_client.alpha.admin.list_routes(api_filter="v1")
        assert routes is not None
        assert len(routes) > 0

        # Should not include deprecated routes
        openai_routes = [r for r in routes if "/openai/" in r.route]
        assert len(openai_routes) == 0

        # Should include v1 routes
        paths = [r.route for r in routes]
        assert any(
            "/v1/" in p or p.startswith("/inspect/") or p.startswith("/health") or p.startswith("/version")
            for p in paths
        )
