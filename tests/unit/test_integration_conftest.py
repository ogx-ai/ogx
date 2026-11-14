# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""tests/integration/conftest.py used to export OGX_TEST_STACK_CONFIG_TYPE so a dozen-odd
call sites could tell server mode from library_client mode. #6667 deletes that env var:
is_server_stack_config() replaces the string check, and the two integration test files whose
module-level pytestmark previously read the env var are now skipped by
pytest_collection_modifyitems instead, since a module-level skipif can't see --stack-config.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.integration.conftest import pytest_collection_modifyitems
from tests.integration.stack_config import (
    LIBRARY_CLIENT_ONLY_TEST_PATHS,
    SERVER_ONLY_TEST_PATHS,
    is_server_stack_config,
)


class TestIsServerStackConfig:
    @pytest.mark.parametrize(
        "stack_config",
        ["server:ci-tests", "docker:ci-tests", "http://localhost:8321", "https://example.com"],
    )
    def test_recognizes_a_real_server(self, stack_config):
        assert is_server_stack_config(stack_config) is True

    @pytest.mark.parametrize("stack_config", ["ci-tests", "inference=remote::ollama", ""])
    def test_recognizes_an_in_process_library_client(self, stack_config):
        assert is_server_stack_config(stack_config) is False

    def test_none_is_library_client(self):
        """The default when --stack-config is not passed at all."""
        assert is_server_stack_config(None) is False


def _fake_item(rootpath: Path, *rel_parts: str) -> MagicMock:
    item = MagicMock()
    item.fspath = str(rootpath.joinpath(*rel_parts))
    item.add_marker = MagicMock()
    return item


def _fake_config(rootpath: Path, stack_config: str | None) -> SimpleNamespace:
    options = {"--stack-config": stack_config, "--suite": None}
    return SimpleNamespace(rootpath=rootpath, getoption=lambda name, default=None: options.get(name, default))


class TestCollectionSkipsTheStackConfigGatedFiles:
    """The two files SERVER_ONLY_TEST_PATHS / LIBRARY_CLIENT_ONLY_TEST_PATHS name are the
    same ones whose old pytestmark read OGX_TEST_STACK_CONFIG_TYPE."""

    def test_server_only_file_is_skipped_in_library_client_mode(self, tmp_path):
        item = _fake_item(tmp_path, *next(iter(SERVER_ONLY_TEST_PATHS)))

        pytest_collection_modifyitems(_fake_config(tmp_path, "ci-tests"), [item])

        item.add_marker.assert_called_once()

    def test_server_only_file_runs_in_server_mode(self, tmp_path):
        item = _fake_item(tmp_path, *next(iter(SERVER_ONLY_TEST_PATHS)))

        pytest_collection_modifyitems(_fake_config(tmp_path, "server:ci-tests"), [item])

        item.add_marker.assert_not_called()

    def test_library_client_only_file_is_skipped_in_server_mode(self, tmp_path):
        item = _fake_item(tmp_path, *next(iter(LIBRARY_CLIENT_ONLY_TEST_PATHS)))

        pytest_collection_modifyitems(_fake_config(tmp_path, "server:ci-tests"), [item])

        item.add_marker.assert_called_once()

    def test_library_client_only_file_runs_in_library_client_mode(self, tmp_path):
        item = _fake_item(tmp_path, *next(iter(LIBRARY_CLIENT_ONLY_TEST_PATHS)))

        pytest_collection_modifyitems(_fake_config(tmp_path, "ci-tests"), [item])

        item.add_marker.assert_not_called()

    def test_an_unrelated_file_is_never_touched(self, tmp_path):
        item = _fake_item(tmp_path, "tests/integration/inference/test_openai_completion.py")

        pytest_collection_modifyitems(_fake_config(tmp_path, "server:ci-tests"), [item])
        pytest_collection_modifyitems(_fake_config(tmp_path, "ci-tests"), [item])

        item.add_marker.assert_not_called()

    def test_an_item_outside_rootpath_is_never_touched(self, tmp_path):
        """item.fspath.relative_to(rootpath) raises for a path outside rootpath (e.g. a plugin
        or conftest collected from elsewhere); that must not crash collection."""
        item = MagicMock()
        item.fspath = str(tmp_path.parent / "somewhere-else" / "test_unrelated.py")
        item.add_marker = MagicMock()

        pytest_collection_modifyitems(_fake_config(tmp_path, "server:ci-tests"), [item])

        item.add_marker.assert_not_called()
