# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path

OGX_CONFIG_DIR = Path(os.getenv("OGX_CONFIG_DIR", os.path.expanduser("~/.ogx/")))

DISTRIBS_BASE_DIR = OGX_CONFIG_DIR / "distributions"

DEFAULT_CHECKPOINT_DIR = OGX_CONFIG_DIR / "checkpoints"

RUNTIME_BASE_DIR = OGX_CONFIG_DIR / "runtime"

EXTERNAL_PROVIDERS_DIR = OGX_CONFIG_DIR / "providers.d"
