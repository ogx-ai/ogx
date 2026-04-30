# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import shutil
import warnings
from pathlib import Path

OGX_CONFIG_DIR = Path(os.getenv("OGX_CONFIG_DIR", os.path.expanduser("~/.ogx/")))

DISTRIBS_BASE_DIR = OGX_CONFIG_DIR / "distributions"

DEFAULT_CHECKPOINT_DIR = OGX_CONFIG_DIR / "checkpoints"

RUNTIME_BASE_DIR = OGX_CONFIG_DIR / "runtime"

EXTERNAL_PROVIDERS_DIR = OGX_CONFIG_DIR / "providers.d"

UI_LOGS_DIR = OGX_CONFIG_DIR / "ui" / "logs"

_LEGACY_CONFIG_DIR = Path(os.path.expanduser("~/.llama"))


def migrate_legacy_config_dir() -> None:
    if os.getenv("OGX_CONFIG_DIR"):
        return

    if not _LEGACY_CONFIG_DIR.exists():
        return

    if OGX_CONFIG_DIR.exists():
        warnings.warn(
            f"Legacy config directory {_LEGACY_CONFIG_DIR} found alongside {OGX_CONFIG_DIR}. "
            f"Please remove legacy directory {_LEGACY_CONFIG_DIR} — it is no longer used.",
            DeprecationWarning,
            stacklevel=2,
        )
        return

    try:
        shutil.move(str(_LEGACY_CONFIG_DIR), str(OGX_CONFIG_DIR))
    except OSError:
        warnings.warn(
            f"Failed to migrate {_LEGACY_CONFIG_DIR} to {OGX_CONFIG_DIR}. Please rename it manually.",
            DeprecationWarning,
            stacklevel=2,
        )
