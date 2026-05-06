# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for `ogx run` CLI command.

Categories:
  - Arguments: --providers flag is registered and parsed correctly
  - Delegation: --providers delegates to run_config_from_dynamic_config_spec
  - Error propagation: ValueError from the unified impl is printed and causes exit
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ogx.cli.stack.run import StackRun


@pytest.fixture
def stack_run() -> StackRun:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return StackRun(subparsers)


class TestArguments:
    def test_providers_flag_registered(self, stack_run: StackRun):
        args = stack_run.parser.parse_args(["--providers", "inference=fireworks"])
        assert args.providers == "inference=fireworks"

    def test_providers_default_is_none(self, stack_run: StackRun):
        args = stack_run.parser.parse_args([])
        assert args.providers is None

    def test_providers_accepts_multiple_pairs(self, stack_run: StackRun):
        args = stack_run.parser.parse_args(["--providers", "inference=fireworks,safety=llama-guard"])
        assert args.providers == "inference=fireworks,safety=llama-guard"


class TestDelegation:
    def test_providers_calls_dynamic_config_spec(self, stack_run: StackRun, tmp_path: Path):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {}

        with (
            patch("ogx.cli.stack.run.run_config_from_dynamic_config_spec", return_value=mock_config) as mock_fn,
            patch("ogx.cli.stack.run.DISTRIBS_BASE_DIR", tmp_path),
            patch(
                "ogx.core.configure.parse_and_maybe_upgrade_config",
                return_value=MagicMock(external_providers_dir=None),
            ),
            patch.object(stack_run, "_uvicorn_run"),
        ):
            args = stack_run.parser.parse_args(["--providers", "inference=fireworks"])
            stack_run._run_stack_run_cmd(args)

        mock_fn.assert_called_once_with(
            dynamic_config_spec="inference=fireworks",
            distro_dir=tmp_path / "providers-run",
            distro_name="providers-run",
        )

    def test_providers_writes_config_yaml(self, stack_run: StackRun, tmp_path: Path):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {"distro_name": "providers-run"}

        with (
            patch("ogx.cli.stack.run.run_config_from_dynamic_config_spec", return_value=mock_config),
            patch("ogx.cli.stack.run.DISTRIBS_BASE_DIR", tmp_path),
            patch(
                "ogx.core.configure.parse_and_maybe_upgrade_config",
                return_value=MagicMock(external_providers_dir=None),
            ),
            patch.object(stack_run, "_uvicorn_run"),
        ):
            args = stack_run.parser.parse_args(["--providers", "inference=fireworks"])
            stack_run._run_stack_run_cmd(args)

        config_file = tmp_path / "providers-run" / "config.yaml"
        assert config_file.exists()


class TestErrorPropagation:
    def test_value_error_causes_exit(self, stack_run: StackRun, tmp_path: Path):
        with (
            patch(
                "ogx.cli.stack.run.run_config_from_dynamic_config_spec",
                side_effect=ValueError("Failed to parse provider spec 'bad'. Expected format: api=provider"),
            ),
            patch("ogx.cli.stack.run.DISTRIBS_BASE_DIR", tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            args = stack_run.parser.parse_args(["--providers", "bad"])
            stack_run._run_stack_run_cmd(args)

        assert exc_info.value.code == 1
