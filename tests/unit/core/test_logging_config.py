# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Regression tests: ensure logging behaves consistently whether the server
is started via `ogx run` or directly via `uvicorn create_app`.

These tests verify that:
1. setup_logging() is called when creating the app
2. Application loggers are properly configured
3. Custom log levels from config.yaml are respected
"""

import logging  # allow-direct-logging
import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from ogx.log import LoggingConfig, _reset_logging_state, parse_yaml_config, setup_logging


@pytest.fixture(autouse=True)
def _clean_logging_state():
    """Reset logging state before and after each test to prevent leakage."""
    _reset_logging_state()
    yield
    _reset_logging_state()


class TestLoggingConfiguration:
    """Test logging configuration setup"""

    def test_parse_yaml_config_with_custom_levels(self):
        """Test that custom log levels from YAML are correctly parsed"""
        # Arrange
        logger_config = LoggingConfig(
            category_levels={
                "core": "DEBUG",
                "server": "WARNING",
                "router": "INFO",
            }
        )

        # Act
        category_levels = parse_yaml_config(logger_config)

        # Assert
        assert "core" in category_levels
        assert category_levels["core"] == logging.DEBUG
        assert "server" in category_levels
        assert category_levels["server"] == logging.WARNING
        assert "router" in category_levels
        assert category_levels["router"] == logging.INFO

    def test_parse_yaml_config_with_all_category(self):
        """Test that 'all' category applies to all loggers"""
        # Arrange
        logger_config = LoggingConfig(category_levels={"all": "DEBUG"})

        # Act
        category_levels = parse_yaml_config(logger_config)

        # Assert
        # Should set all known categories plus root
        assert "core" in category_levels
        assert "server" in category_levels
        assert "router" in category_levels
        assert "root" in category_levels
        assert category_levels["core"] == logging.DEBUG
        assert category_levels["root"] == logging.DEBUG

    def test_setup_logging_creates_loggers(self):
        """Test that setup_logging() properly configures loggers"""
        # Arrange
        category_levels = {
            "core": logging.DEBUG,
            "server": logging.INFO,
        }

        # Act
        setup_logging(category_levels)

        # Assert - loggers should be configured (have handlers)
        assert len(logging.root.handlers) > 0

    def test_setup_logging_with_no_config(self):
        """Test that setup_logging() works with no custom config (defaults)"""
        # Act - should not raise an exception
        setup_logging()

        # Assert - loggers should still be created
        assert len(logging.root.handlers) > 0

    @patch("ogx.core.server.server.setup_logging")
    def test_create_app_calls_setup_logging_with_config(self, mock_setup_logging):
        """
        Verify that create_app() calls setup_logging() when
        logging_config is present in config file.
        """
        # Arrange - create a temporary config file with logging_config
        config_data = {
            "server": {
                "port": 8000,
                "host": "127.0.0.1",
            },
            "logging_config": {
                "category_levels": {
                    "core": "DEBUG",
                    "server": "INFO",
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Set environment variable
            os.environ["OGX_CONFIG"] = config_file

            # Act - import here to avoid early initialization
            from ogx.core.server.server import create_app

            # This will fail due to missing Stack implementation, but we can verify
            # that setup_logging was called
            try:
                create_app()
            except Exception:
                # Expected to fail due to missing Stack, but setup_logging should have been called
                pass

            # Assert - verify setup_logging was called with correct category_levels
            assert mock_setup_logging.called
            call_args = mock_setup_logging.call_args
            if call_args[0]:  # positional args
                category_levels = call_args[0][0]
                assert "core" in category_levels
                assert category_levels["core"] == logging.DEBUG
        finally:
            # Cleanup
            os.unlink(config_file)
            if "OGX_CONFIG" in os.environ:
                del os.environ["OGX_CONFIG"]

    @patch("ogx.core.server.server.setup_logging")
    def test_create_app_calls_setup_logging_without_config(self, mock_setup_logging):
        """
        Verify that create_app() calls setup_logging() even when
        logging_config is NOT present in config file.
        """
        # Arrange - create a temporary config file WITHOUT logging_config
        config_data = {
            "server": {
                "port": 8000,
                "host": "127.0.0.1",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Set environment variable
            os.environ["OGX_CONFIG"] = config_file

            # Act
            from ogx.core.server.server import create_app

            try:
                create_app()
            except Exception:
                # Expected to fail due to missing Stack, but setup_logging should have been called
                pass

            # Assert - verify setup_logging was called (with no args or None)
            assert mock_setup_logging.called
        finally:
            # Cleanup
            os.unlink(config_file)
            if "OGX_CONFIG" in os.environ:
                del os.environ["OGX_CONFIG"]


class TestLoggingConfigIntegration:
    """Integration tests for logging configuration with actual logging output"""

    def test_ogx_loggers_are_configured(self):
        """Test that ogx.* loggers are properly configured after setup_logging()"""
        # Arrange
        category_levels = {"core": logging.DEBUG}
        setup_logging(category_levels)

        # Act - create various ogx loggers
        loggers = [
            logging.getLogger("ogx.core.server"),
            logging.getLogger("ogx.core.stack"),
            logging.getLogger("ogx.cli.stack.run"),
        ]

        # Assert - all loggers should have handlers (configured)
        for logger in loggers:
            # Walk up the logger hierarchy to find handlers
            current = logger
            has_handler = False
            while current:
                if current.handlers:
                    has_handler = True
                    break
                if not current.propagate:
                    break
                current = current.parent

            assert has_handler or len(logging.root.handlers) > 0, f"Logger {logger.name} has no handlers"

    def test_custom_log_levels_are_applied(self):
        """Test that custom log levels from config are actually applied to loggers"""
        # Arrange
        from ogx.log import get_logger

        category_levels = {
            "core": logging.DEBUG,
            "server": logging.WARNING,
            "router": logging.INFO,
        }
        setup_logging(category_levels)

        # Act - get loggers for each category (calls register the stdlib logger + level)
        get_logger("test.core", category="core")
        get_logger("test.server", category="server")
        get_logger("test.router", category="router")

        # Assert - verify effective log levels on the underlying stdlib loggers
        # get_logger returns a structlog BoundLogger; check the stdlib logger directly
        assert logging.getLogger("test.core").level == logging.DEBUG
        assert logging.getLogger("test.server").level == logging.WARNING
        assert logging.getLogger("test.router").level == logging.INFO

    def test_setup_logging_updates_preexisting_loggers(self):
        """
        Regression test: loggers created before setup_logging() (e.g. at module
        import time) must have their levels updated when setup_logging() is
        called later with custom category levels.
        """
        from ogx.log import get_logger

        # Arrange - reset to defaults, then create loggers BEFORE calling
        # setup_logging with custom levels (simulates module-level imports)
        setup_logging({"core": logging.INFO})
        get_logger("test.preexisting.auth", category="core::auth")
        get_logger("test.preexisting.server", category="core::server")
        assert logging.getLogger("test.preexisting.auth").level == logging.INFO
        assert logging.getLogger("test.preexisting.server").level == logging.INFO

        # Act - call setup_logging with custom levels, as create_app() does
        setup_logging({"core": logging.DEBUG})

        # Assert - pre-existing loggers must now reflect the updated level
        assert logging.getLogger("test.preexisting.auth").level == logging.DEBUG
        assert logging.getLogger("test.preexisting.server").level == logging.DEBUG
