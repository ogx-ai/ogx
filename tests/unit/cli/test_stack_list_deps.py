# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for `ogx list-deps` CLI command.

Categories:
  - Arguments: --providers flag is registered and parsed correctly
  - Delegation: --providers delegates to run_config_from_dynamic_config_spec rather than its own parsing loop
  - Error propagation: ValueError from the unified impl is printed and causes exit(1)
  - Output: dependencies from get_provider_dependencies are printed to stdout
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from ogx.cli.stack.list_deps import StackListDeps


@pytest.fixture
def stack_list_deps() -> StackListDeps:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return StackListDeps(subparsers)


class TestArguments:
    def test_providers_flag_registered(self, stack_list_deps: StackListDeps):
        args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks"])
        assert args.providers == "inference=fireworks"

    def test_providers_default_is_none(self, stack_list_deps: StackListDeps):
        args = stack_list_deps.parser.parse_args([])
        assert args.providers is None

    def test_providers_accepts_multiple_pairs(self, stack_list_deps: StackListDeps):
        args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks,safety=llama-guard"])
        assert args.providers == "inference=fireworks,safety=llama-guard"

    def test_config_and_providers_are_independent(self, stack_list_deps: StackListDeps):
        # --providers with no positional config
        args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks"])
        assert args.config is None
        assert args.providers == "inference=fireworks"


class TestDelegation:
    def _mock_provider_registry(self):
        """Return a mock registry that accepts test provider types."""
        from ogx.core.datatypes import Api

        return {
            Api.inference: {"fireworks": MagicMock()},
            Api.safety: {"llama-guard": MagicMock()},
        }

    def test_providers_calls_dynamic_config_spec(self, stack_list_deps: StackListDeps):
        mock_config = MagicMock()
        mock_config.external_apis_dir = None

        with (
            patch(
                "ogx.cli.stack._list_deps.run_config_from_dynamic_config_spec",
                return_value=mock_config,
            ) as mock_fn,
            patch(
                "ogx.cli.stack._list_deps.get_provider_registry",
                return_value=self._mock_provider_registry(),
            ),
            patch(
                "ogx.cli.stack._list_deps.add_dependent_providers",
            ),
            patch(
                "ogx.cli.stack._list_deps.get_provider_dependencies",
                return_value=([], [], []),
            ),
            patch("builtins.print"),
        ):
            args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks"])
            stack_list_deps._run_stack_list_deps_command(args)

        mock_fn.assert_called_once_with("inference=fireworks")

    def test_providers_passes_semicolon_spec_unchanged(self, stack_list_deps: StackListDeps):
        # The function accepts both comma- and semicolon-separated specs; the unified
        # parser is responsible for normalisation — list-deps should forward verbatim.
        mock_config = MagicMock()
        mock_config.external_apis_dir = None

        with (
            patch(
                "ogx.cli.stack._list_deps.run_config_from_dynamic_config_spec",
                return_value=mock_config,
            ) as mock_fn,
            patch(
                "ogx.cli.stack._list_deps.get_provider_registry",
                return_value=self._mock_provider_registry(),
            ),
            patch(
                "ogx.cli.stack._list_deps.add_dependent_providers",
            ),
            patch(
                "ogx.cli.stack._list_deps.get_provider_dependencies",
                return_value=([], [], []),
            ),
            patch("builtins.print"),
        ):
            args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks;safety=llama-guard"])
            stack_list_deps._run_stack_list_deps_command(args)

        mock_fn.assert_called_once_with("inference=fireworks;safety=llama-guard")


class TestErrorPropagation:
    def test_value_error_causes_exit_1(self, stack_list_deps: StackListDeps):
        with (
            patch(
                "ogx.cli.stack._list_deps.run_config_from_dynamic_config_spec",
                side_effect=ValueError("Failed to parse provider spec 'bad'"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            args = stack_list_deps.parser.parse_args(["--providers", "bad"])
            stack_list_deps._run_stack_list_deps_command(args)

        assert exc_info.value.code == 1

    def test_value_error_message_printed_to_stderr(self, stack_list_deps: StackListDeps, capsys):
        with (
            patch(
                "ogx.cli.stack._list_deps.run_config_from_dynamic_config_spec",
                side_effect=ValueError("Failed to parse provider spec 'bad'"),
            ),
            pytest.raises(SystemExit),
        ):
            args = stack_list_deps.parser.parse_args(["--providers", "bad"])
            stack_list_deps._run_stack_list_deps_command(args)

        captured = capsys.readouterr()
        assert "Failed to parse provider spec 'bad'" in captured.err


class TestOutput:
    def _mock_provider_registry(self):
        """Return a mock registry that accepts test provider types."""
        from ogx.core.datatypes import Api

        return {
            Api.inference: {"fireworks": MagicMock()},
        }

    def test_normal_deps_printed(self, stack_list_deps: StackListDeps, capsys):
        mock_config = MagicMock()
        mock_config.external_apis_dir = None

        with (
            patch(
                "ogx.cli.stack._list_deps.run_config_from_dynamic_config_spec",
                return_value=mock_config,
            ),
            patch(
                "ogx.cli.stack._list_deps.get_provider_registry",
                return_value=self._mock_provider_registry(),
            ),
            patch(
                "ogx.cli.stack._list_deps.add_dependent_providers",
            ),
            patch(
                "ogx.cli.stack._list_deps.get_provider_dependencies",
                return_value=(["httpx", "aiohttp"], [], []),
            ),
        ):
            args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks"])
            stack_list_deps._run_stack_list_deps_command(args)

        output = capsys.readouterr().out
        assert "httpx" in output
        assert "aiohttp" in output

    def test_server_dependencies_always_included(self, stack_list_deps: StackListDeps, capsys):
        mock_config = MagicMock()
        mock_config.external_apis_dir = None

        with (
            patch(
                "ogx.cli.stack._list_deps.run_config_from_dynamic_config_spec",
                return_value=mock_config,
            ),
            patch(
                "ogx.cli.stack._list_deps.get_provider_registry",
                return_value=self._mock_provider_registry(),
            ),
            patch(
                "ogx.cli.stack._list_deps.add_dependent_providers",
            ),
            patch(
                "ogx.cli.stack._list_deps.get_provider_dependencies",
                return_value=([], [], []),
            ),
        ):
            args = stack_list_deps.parser.parse_args(["--providers", "inference=fireworks"])
            stack_list_deps._run_stack_list_deps_command(args)

        output = capsys.readouterr().out
        # SERVER_DEPENDENCIES are always appended
        assert "fastapi" in output
        assert "uvicorn" in output
