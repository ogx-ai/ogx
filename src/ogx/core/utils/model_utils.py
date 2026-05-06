# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from .config_dirs import DEFAULT_CHECKPOINT_DIR


def model_local_dir(descriptor: str) -> str:
    """Get the local directory path for storing model checkpoints.

    Args:
        descriptor: The model descriptor string (colons are replaced with dashes).

    Returns:
        The absolute path to the model's local checkpoint directory.
    """
    return str(Path(DEFAULT_CHECKPOINT_DIR) / (descriptor.replace(":", "-")))
