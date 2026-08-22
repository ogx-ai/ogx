# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for `ogx run` and `ogx stack run` CLI commands.

Categories:
  - Arguments: --providers and --dry-run flags are registered and parsed correctly
  - Delegation: --providers delegates to run_config_from_dynamic_config_spec
  - Error propagation: ValueError from the unified impl is printed and causes exit
  - Deprecation: `ogx stack run` emits a FutureWarning
  - Top-level run: `ogx run` works without the `stack` subcommand
  - Dry run: --dry-run validates config without starting the server
"""

import argparse
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ogx.cli.run import Run
from ogx.cli.stack.run import StackRun


@pytest.fixture
def stack_run() -> StackRun:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return StackRun(subparsers)


@pytest.fixture
def top_level_run() -> Run:
    subparsers = argparse.ArgumentParser().add_subparsers()
    return Run(subparsers)


class TestArguments:
    def test_providers_flag_registered(self, stack_run: StackRun):
        args = stack_run.parser.parse_args(["--providers", "inference=fireworks"])
        assert args.providers == "inference=fireworks"

    def test_providers_default_is_none(self, stack_run: StackRun):
        args = stack_run.parser.parse_args([])
        assert args.providers is None

    def test_providers_accepts_multiple_pairs(self, stack_run: StackRun):
        args = stack_run.parser.parse_args(["--providers", "inference=fireworks,responses=builtin"])
        assert args.providers == "inference=fireworks,responses=builtin"


class TestTopLevelRunArguments:
    def test_providers_flag_registered(self, top_level_run: Run):
        args = top_level_run.parser.parse_args(["--providers", "inference=fireworks"])
        assert args.providers == "inference=fireworks"

    def test_providers_default_is_none(self, top_level_run: Run):
        args = top_level_run.parser.parse_args([])
        assert args.providers is None

    def test_config_positional(self, top_level_run: Run):
        args = top_level_run.parser.parse_args(["starter"])
        assert args.config == "starter"

    def test_all_options(self, top_level_run: Run):
        args = top_level_run.parser.parse_args(["starter", "--port", "9000", "--enable-ui"])
        assert args.config == "starter"
        assert args.port == 9000
        assert args.enable_ui is True


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
            patch("ogx.cli.stack.run._uvicorn_run"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
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
            patch("ogx.cli.stack.run._uvicorn_run"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            args = stack_run.parser.parse_args(["--providers", "inference=fireworks"])
            stack_run._run_stack_run_cmd(args)

        config_file = tmp_path / "providers-run" / "config.yaml"
        assert config_file.exists()

    def test_top_level_run_delegates(self, top_level_run: Run, tmp_path: Path):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {}

        with (
            patch("ogx.cli.stack.run.run_config_from_dynamic_config_spec", return_value=mock_config) as mock_fn,
            patch("ogx.cli.stack.run.DISTRIBS_BASE_DIR", tmp_path),
            patch(
                "ogx.core.configure.parse_and_maybe_upgrade_config",
                return_value=MagicMock(external_providers_dir=None),
            ),
            patch("ogx.cli.stack.run._uvicorn_run"),
        ):
            args = top_level_run.parser.parse_args(["--providers", "inference=fireworks"])
            top_level_run._run_cmd(args)

        mock_fn.assert_called_once_with(
            dynamic_config_spec="inference=fireworks",
            distro_dir=tmp_path / "providers-run",
            distro_name="providers-run",
        )


class TestErrorPropagation:
    def test_value_error_causes_exit(self, stack_run: StackRun, tmp_path: Path):
        with (
            patch(
                "ogx.cli.stack.run.run_config_from_dynamic_config_spec",
                side_effect=ValueError("Failed to parse provider spec 'bad'. Expected format: api=provider"),
            ),
            patch("ogx.cli.stack.run.DISTRIBS_BASE_DIR", tmp_path),
            warnings.catch_warnings(),
            pytest.raises(SystemExit) as exc_info,
        ):
            warnings.simplefilter("ignore", FutureWarning)
            args = stack_run.parser.parse_args(["--providers", "bad"])
            stack_run._run_stack_run_cmd(args)

        assert exc_info.value.code == 1


class TestDeprecation:
    def test_stack_run_emits_deprecation_warning(self, stack_run: StackRun, tmp_path: Path):
        with (
            patch("ogx.cli.stack.run.run_stack_cmd"),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            args = stack_run.parser.parse_args(["starter"])
            stack_run._run_stack_run_cmd(args)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1
        assert "deprecated" in str(future_warnings[0].message)
        assert "ogx run" in str(future_warnings[0].message)

    def test_top_level_run_no_deprecation_warning(self, top_level_run: Run, tmp_path: Path):
        with (
            patch("ogx.cli.run.run_stack_cmd"),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            args = top_level_run.parser.parse_args(["starter"])
            top_level_run._run_cmd(args)

        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 0


class TestDryRun:
    def test_dry_run_flag_registered(self, top_level_run: Run):
        args = top_level_run.parser.parse_args(["starter", "--dry-run"])
        assert args.dry_run is True

    def test_dry_run_default_is_false(self, top_level_run: Run):
        args = top_level_run.parser.parse_args(["starter"])
        assert args.dry_run is False

    def test_dry_run_validates_and_exits_without_starting_server(self, top_level_run: Run, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("version: 2\ndistro_name: test\napis: []\nproviders: {}\n")

        mock_config = MagicMock(external_providers_dir=None)

        with (
            patch("ogx.core.configure.parse_and_maybe_upgrade_config", return_value=mock_config),
            patch("ogx.cli.stack.run._dry_run_validate") as mock_validate,
            patch("ogx.cli.stack.run._uvicorn_run") as mock_uvicorn,
            patch("ogx.cli.stack.run.resolve_config_or_distro", return_value=config_file),
        ):
            args = top_level_run.parser.parse_args([str(config_file), "--dry-run"])
            top_level_run._run_cmd(args)

        mock_validate.assert_called_once_with(mock_config, config_file)
        mock_uvicorn.assert_not_called()

    def test_dry_run_without_config_errors(self, top_level_run: Run):
        with pytest.raises(SystemExit):
            args = top_level_run.parser.parse_args(["--dry-run"])
            top_level_run._run_cmd(args)

    def test_dry_run_with_providers_flag(self, top_level_run: Run, tmp_path: Path):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {}

        with (
            patch("ogx.cli.stack.run.run_config_from_dynamic_config_spec", return_value=mock_config),
            patch("ogx.cli.stack.run.DISTRIBS_BASE_DIR", tmp_path),
            patch(
                "ogx.core.configure.parse_and_maybe_upgrade_config",
                return_value=MagicMock(external_providers_dir=None),
            ),
            patch("ogx.cli.stack.run._dry_run_validate") as mock_validate,
            patch("ogx.cli.stack.run._uvicorn_run") as mock_uvicorn,
        ):
            args = top_level_run.parser.parse_args(["--providers", "inference=fireworks", "--dry-run"])
            top_level_run._run_cmd(args)

        mock_validate.assert_called_once()
        mock_uvicorn.assert_not_called()
