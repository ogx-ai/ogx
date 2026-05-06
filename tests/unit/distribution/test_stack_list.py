# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the ogx list command."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from ogx.cli.stack.list_stacks import StackListBuilds


@pytest.fixture
def list_stacks_command():
    """Create a StackListBuilds instance for testing."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    return StackListBuilds(subparsers)


@pytest.fixture
def mock_distribs_base_dir(tmp_path):
    """Create a mock DISTRIBS_BASE_DIR with some custom distributions."""
    custom_dir = tmp_path / "distributions"
    custom_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom distribution
    starter_custom = custom_dir / "starter"
    starter_custom.mkdir()
    (starter_custom / "starter-config.yaml").write_text("# config")

    return custom_dir


@pytest.fixture
def mock_distro_dir(tmp_path):
    """Create a mock distributions directory with built-in distributions."""
    distro_dir = tmp_path / "src" / "ogx" / "distributions"
    distro_dir.mkdir(parents=True, exist_ok=True)

    # Create some built-in distributions
    for distro_name in ["starter", "nvidia"]:
        distro_path = distro_dir / distro_name
        distro_path.mkdir()
        (distro_path / "config.yaml").write_text("# config")

    return distro_dir


def create_path_mock(builtin_dist_dir):
    """Create a properly mocked Path object that returns builtin_dist_dir for the distributions path."""
    mock_parent_parent_parent = MagicMock()
    mock_parent_parent_parent.__truediv__ = lambda self, other: (
        builtin_dist_dir if other == "distributions" else MagicMock()
    )

    mock_path = MagicMock()
    mock_path.parent.parent.parent = mock_parent_parent_parent

    return mock_path


class TestStackList:
    """Test suite for ogx list command."""

    def test_builtin_distros_shown_without_running(self, list_stacks_command, mock_distro_dir, tmp_path):
        """Test that built-in distributions are shown even before running them."""
        mock_path = create_path_mock(mock_distro_dir)

        # Mock DISTRIBS_BASE_DIR to be a non-existent directory (no custom distributions)
        with patch("ogx.cli.stack.list_stacks.DISTRIBS_BASE_DIR", tmp_path / "nonexistent"):
            with patch("ogx.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # Verify built-in distributions are found
                assert len(distributions) > 0, "Should find built-in distributions"
                assert all(source_type == "built-in" for _, source_type in distributions.values()), (
                    "All should be built-in"
                )

                # Check specific distributions we created
                assert "starter" in distributions
                assert "nvidia" in distributions

    def test_custom_distribution_overrides_builtin(self, list_stacks_command, mock_distro_dir, mock_distribs_base_dir):
        """Test that custom distributions override built-in ones with the same name."""
        mock_path = create_path_mock(mock_distro_dir)

        with patch("ogx.cli.stack.list_stacks.DISTRIBS_BASE_DIR", mock_distribs_base_dir):
            with patch("ogx.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # "starter" should exist and be marked as "custom" (not "built-in")
                # because the custom version overrides the built-in one
                assert "starter" in distributions
                _, source_type = distributions["starter"]
                assert source_type == "custom", "Custom distribution should override built-in"

    def test_hidden_directories_ignored(self, list_stacks_command, mock_distro_dir, tmp_path):
        """Test that hidden directories (starting with .) are ignored."""
        # Add a hidden directory
        hidden_dir = mock_distro_dir / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "config.yaml").write_text("# config")

        # Add a __pycache__ directory
        pycache_dir = mock_distro_dir / "__pycache__"
        pycache_dir.mkdir()

        mock_path = create_path_mock(mock_distro_dir)

        with patch("ogx.cli.stack.list_stacks.DISTRIBS_BASE_DIR", tmp_path / "nonexistent"):
            with patch("ogx.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                assert ".hidden" not in distributions
                assert "__pycache__" not in distributions
