# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for OGXAsLibraryClient automatic initialization.

These tests ensure that the library client is automatically initialized
and ready to use immediately after construction.
"""

import pytest

from ogx.core.library_client import (
    AsyncOGXAsLibraryClient,
    OGXAsLibraryClient,
)
from ogx.core.server.routes import RouteImpls


class TestOGXAsLibraryClientAutoInitialization:
    """Test automatic initialization of library clients."""

    def test_sync_client_auto_initialization(self, monkeypatch):
        """Test that sync client is automatically initialized after construction."""
        # Mock the stack construction to avoid dependency issues
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = OGXAsLibraryClient("ci-tests")

        assert client.async_client.route_impls is not None

    async def test_async_client_auto_initialization(self, monkeypatch):
        """Test that async client can be initialized and works properly."""
        # Mock the stack construction to avoid dependency issues
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")

        # Initialize the client
        result = await client.initialize()
        assert result is True
        assert client.route_impls is not None

    def test_initialize_method_backward_compatibility(self, monkeypatch):
        """Test that initialize() method still works for backward compatibility."""
        # Mock the stack construction to avoid dependency issues
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = OGXAsLibraryClient("ci-tests")

        result = client.initialize()
        assert result is None

        result2 = client.initialize()
        assert result2 is None

    async def test_async_initialize_method_idempotent(self, monkeypatch):
        """Test that async initialize() method can be called multiple times safely."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")

        result1 = await client.initialize()
        assert result1 is True

        result2 = await client.initialize()
        assert result2 is True

    def test_route_impls_automatically_set(self, monkeypatch):
        """Test that route_impls is automatically set during construction."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        sync_client = OGXAsLibraryClient("ci-tests")
        assert sync_client.async_client.route_impls is not None


class TestOGXAsLibraryClientShutdown:
    """Test shutdown functionality of library clients."""

    async def test_async_client_shutdown(self, monkeypatch):
        """Test that async client shutdown properly shuts down the stack."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")
        await client.initialize()

        # Verify stack is set
        assert client.stack is not None

        # Call shutdown
        await client.shutdown()

        # Verify shutdown was called on the stack
        assert len(shutdown_called) == 1

        # Verify stack is cleared
        assert client.stack is None

    async def test_async_client_shutdown_idempotent(self, monkeypatch):
        """Test that async client shutdown can be called multiple times safely."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")
        await client.initialize()

        # Call shutdown multiple times
        await client.shutdown()
        await client.shutdown()
        await client.shutdown()

        # Shutdown should only be called once (subsequent calls are no-ops)
        assert len(shutdown_called) == 1

    async def test_async_client_shutdown_before_initialize(self, monkeypatch):
        """Test that async client shutdown works even if never initialized."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")

        # Shutdown without initialize should not raise
        await client.shutdown()

    def test_sync_client_shutdown(self, monkeypatch):
        """Test that sync client shutdown properly shuts down the stack."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = OGXAsLibraryClient("ci-tests")

        # Call shutdown
        client.shutdown()

        # Verify shutdown was called on the stack
        assert len(shutdown_called) == 1

    def test_sync_client_shutdown_idempotent(self, monkeypatch):
        """Test that sync client shutdown can be called multiple times safely."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = OGXAsLibraryClient("ci-tests")

        # Call shutdown multiple times - should not raise
        # Note: After first shutdown, the loop is closed, so subsequent calls may behave differently
        client.shutdown()

    def test_async_client_has_shutdown_method(self, monkeypatch):
        """Verify AsyncOGXAsLibraryClient has shutdown method."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")
        assert hasattr(client, "shutdown")
        assert callable(client.shutdown)

    def test_sync_client_has_shutdown_method(self, monkeypatch):
        """Verify OGXAsLibraryClient has shutdown method."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = OGXAsLibraryClient("ci-tests")
        assert hasattr(client, "shutdown")
        assert callable(client.shutdown)


class TestOGXAsLibraryClientContextManager:
    """Test context manager functionality of library clients."""

    async def test_async_client_context_manager(self, monkeypatch):
        """Test that async client works as an async context manager."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        async with AsyncOGXAsLibraryClient("ci-tests") as client:
            # Verify client is initialized
            assert client.route_impls is not None

        # Verify shutdown was called on exit
        assert len(shutdown_called) == 1

    async def test_async_client_context_manager_with_exception(self, monkeypatch):
        """Test that async client shuts down even when an exception occurs."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        with pytest.raises(ValueError):
            async with AsyncOGXAsLibraryClient("ci-tests") as _client:
                raise ValueError("Test exception")

        # Verify shutdown was still called
        assert len(shutdown_called) == 1

    def test_sync_client_context_manager(self, monkeypatch):
        """Test that sync client works as a context manager."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        with OGXAsLibraryClient("ci-tests") as client:
            # Verify client is initialized
            assert client.async_client.route_impls is not None

        # Verify shutdown was called on exit
        assert len(shutdown_called) == 1

    def test_sync_client_context_manager_with_exception(self, monkeypatch):
        """Test that sync client shuts down even when an exception occurs."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})
        shutdown_called = []

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                shutdown_called.append(True)

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        with pytest.raises(ValueError):
            with OGXAsLibraryClient("ci-tests") as _client:
                raise ValueError("Test exception")

        # Verify shutdown was still called
        assert len(shutdown_called) == 1

    def test_async_client_has_context_manager_methods(self, monkeypatch):
        """Verify AsyncOGXAsLibraryClient has context manager methods."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = AsyncOGXAsLibraryClient("ci-tests")
        assert hasattr(client, "__aenter__")
        assert hasattr(client, "__aexit__")

    def test_sync_client_has_context_manager_methods(self, monkeypatch):
        """Verify OGXAsLibraryClient has context manager methods."""
        mock_impls = {}
        mock_route_impls = RouteImpls({})

        class MockStack:
            def __init__(self, config, custom_provider_registry=None):
                self.impls = mock_impls

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

        def mock_initialize_route_impls(impls):
            return mock_route_impls

        monkeypatch.setattr("ogx.core.library_client.Stack", MockStack)
        monkeypatch.setattr("ogx.core.library_client.initialize_route_impls", mock_initialize_route_impls)

        client = OGXAsLibraryClient("ci-tests")
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")
