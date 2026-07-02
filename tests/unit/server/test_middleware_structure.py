# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
from pathlib import Path

import pytest


class TestMiddlewareModuleStructure:
    """Verify that auth middleware modules are importable from their canonical paths."""

    def test_auth_module_importable(self):
        mod = importlib.import_module("ogx.core.server.middleware.auth")
        assert mod is not None

    def test_auth_providers_module_importable(self):
        mod = importlib.import_module("ogx.core.server.middleware.auth_providers")
        assert mod is not None

    def test_middleware_package_importable(self):
        mod = importlib.import_module("ogx.core.server.middleware")
        assert mod is not None


class TestMiddlewarePublicAPI:
    """Ensure the public API surface of auth middleware is fully accessible from the new location."""

    def test_authentication_middleware_importable(self):
        from ogx.core.server.middleware.auth import AuthenticationMiddleware

        assert AuthenticationMiddleware is not None

    def test_route_authorization_middleware_importable(self):
        from ogx.core.server.middleware.auth import RouteAuthorizationMiddleware

        assert RouteAuthorizationMiddleware is not None

    def test_tenancy_middleware_importable(self):
        from ogx.core.server.middleware.auth import TenancyMiddleware

        assert TenancyMiddleware is not None

    def test_create_auth_provider_importable(self):
        from ogx.core.server.middleware.auth_providers import create_auth_provider

        assert callable(create_auth_provider)

    def test_get_attributes_from_claims_importable(self):
        from ogx.core.server.middleware.auth_providers import get_attributes_from_claims

        assert callable(get_attributes_from_claims)

    @pytest.mark.parametrize(
        "cls_name",
        [
            "OAuth2TokenAuthProvider",
            "CustomAuthProvider",
            "GitHubTokenAuthProvider",
            "KubernetesAuthProvider",
            "UpstreamHeaderAuthProvider",
        ],
    )
    def test_auth_provider_classes_importable(self, cls_name):
        mod = importlib.import_module("ogx.core.server.middleware.auth_providers")
        cls = getattr(mod, cls_name, None)
        assert cls is not None, f"{cls_name} not found in middleware.auth_providers"


class TestMiddlewareDirectoryLayout:
    """Verify the middleware directory has the expected file structure."""

    @pytest.fixture
    def middleware_dir(self):
        import ogx.core.server.middleware as pkg

        return Path(pkg.__file__).parent

    def test_init_exists(self, middleware_dir):
        assert (middleware_dir / "__init__.py").is_file()

    def test_auth_module_exists(self, middleware_dir):
        assert (middleware_dir / "auth.py").is_file()

    def test_auth_providers_module_exists(self, middleware_dir):
        assert (middleware_dir / "auth_providers.py").is_file()

    def test_old_location_removed(self, middleware_dir):
        server_dir = middleware_dir.parent
        assert not (server_dir / "auth.py").exists(), "auth.py should not remain in server/"
        assert not (server_dir / "auth_providers.py").exists(), "auth_providers.py should not remain in server/"


class TestServerImportsMiddleware:
    """Verify that server.py correctly imports middleware from the new location."""

    def test_server_exposes_authentication_middleware(self):
        from ogx.core.server.server import ClientVersionMiddleware

        assert ClientVersionMiddleware is not None

    def test_server_module_loads_without_error(self):
        mod = importlib.import_module("ogx.core.server.server")
        assert hasattr(mod, "create_app")
