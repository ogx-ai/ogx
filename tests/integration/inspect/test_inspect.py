# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import LlamaStackClient

from ogx.core.library_client import OGXAsLibraryClient


class TestInspect:
    @pytest.mark.skip(reason="inspect tests disabled")
    def test_health(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        health = ogx_client.inspect.health()
        assert health is not None
        assert health.status == "OK"

    @pytest.mark.skip(reason="inspect tests disabled")
    def test_version(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        version = ogx_client.inspect.version()
        assert version is not None
        assert version.version is not None

    @pytest.mark.skip(reason="inspect tests disabled")
    def test_list_routes_default(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        """Test list_routes with default filter (non-deprecated v1 routes)."""
        response = ogx_client.routes.list()
        assert response is not None
        assert hasattr(response, "data")
        routes = response.data
        assert len(routes) > 0

        # All routes should be non-deprecated
        # Check that we don't see any /openai/ routes (which are deprecated)
        openai_routes = [r for r in routes if "/openai/" in r.route]
        assert len(openai_routes) == 0, "Default filter should not include deprecated /openai/ routes"

        # Should see standard v1 routes like /inspect/routes, /health, /version
        paths = [r.route for r in routes]
        assert "/inspect/routes" in paths or "/v1/inspect/routes" in paths
        assert "/health" in paths or "/v1/health" in paths

    @pytest.mark.skip(reason="inspect tests disabled")
    def test_list_routes_filter_by_deprecated(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        """Test list_routes with deprecated filter."""
        response = ogx_client.routes.list(api_filter="deprecated")
        assert response is not None
        assert hasattr(response, "data")
        routes = response.data

        # When filtering for deprecated, we should get deprecated routes
        # At minimum, we should see some /openai/ routes which are deprecated
        if len(routes) > 0:
            # If there are any deprecated routes, they should include openai routes
            openai_routes = [r for r in routes if "/openai/" in r.route]
            assert len(openai_routes) > 0, "Deprecated filter should include /openai/ routes"

    @pytest.mark.skip(reason="inspect tests disabled")
    def test_list_routes_filter_by_v1(self, ogx_client: OGXAsLibraryClient | LlamaStackClient):
        """Test list_routes with v1 filter."""
        response = ogx_client.routes.list(api_filter="v1")
        assert response is not None
        assert hasattr(response, "data")
        routes = response.data
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
