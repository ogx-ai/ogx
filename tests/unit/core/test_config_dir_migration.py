# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings

import pytest

from ogx.core.utils.config_dirs import migrate_legacy_config_dir


@pytest.fixture
def config_dirs(tmp_path, monkeypatch):
    """Redirect legacy and new config dirs to tmp_path for isolation."""
    legacy = tmp_path / ".llama"
    new = tmp_path / ".ogx"
    monkeypatch.setattr("ogx.core.utils.config_dirs._LEGACY_CONFIG_DIR", legacy)
    monkeypatch.setattr("ogx.core.utils.config_dirs.OGX_CONFIG_DIR", new)
    monkeypatch.delenv("OGX_CONFIG_DIR", raising=False)
    return legacy, new


def test_migrate_renames_legacy_to_new(config_dirs):
    legacy, new = config_dirs
    legacy.mkdir()
    (legacy / "test_file").write_text("data")

    migrate_legacy_config_dir()

    assert not legacy.exists()
    assert new.exists()
    assert (new / "test_file").read_text() == "data"


def test_warns_when_both_exist(config_dirs):
    legacy, new = config_dirs
    legacy.mkdir()
    new.mkdir()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        migrate_legacy_config_dir()

    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "no longer used" in str(w[0].message)
    assert legacy.exists()
    assert new.exists()


def test_noop_when_neither_exists(config_dirs):
    legacy, new = config_dirs

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        migrate_legacy_config_dir()

    assert len(w) == 0
    assert not legacy.exists()
    assert not new.exists()


def test_skips_when_env_var_set(config_dirs, monkeypatch):
    legacy, new = config_dirs
    legacy.mkdir()
    monkeypatch.setenv("OGX_CONFIG_DIR", "/custom/path")

    migrate_legacy_config_dir()

    assert legacy.exists()
    assert not new.exists()


def test_warns_on_rename_failure(config_dirs, monkeypatch):
    legacy, new = config_dirs
    legacy.mkdir()

    def _fail_move(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr("ogx.core.utils.config_dirs.shutil.move", _fail_move)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        migrate_legacy_config_dir()

    assert len(w) == 1
    assert "Failed to migrate" in str(w[0].message)
    assert legacy.exists()
