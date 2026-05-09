# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")


# TODO: this can probably just be inlined now?
def add_config_distro_args(parser: argparse.ArgumentParser) -> None:
    """Add unified config/distro arguments."""
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "config",
        nargs="?",
        help="Configuration file path or distribution name",
    )


def get_config_from_args(args: argparse.Namespace) -> str | None:
    """Extract the configuration file path from parsed CLI arguments.

    Args:
        args: parsed argparse namespace containing a config attribute.

    Returns:
        The config path as a string, or None if not provided.
    """
    if args.config is not None:
        return str(args.config)
    return None
