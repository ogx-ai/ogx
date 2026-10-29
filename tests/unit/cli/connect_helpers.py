# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.cli.connect.connect import ConnectParser


def parse_group_args(argv: list[str]) -> argparse.Namespace:
    """Parse argv (without the leading 'connect') through the real connect group.

    The group resolves args.connect_url exactly as the production CLI does, so
    per-agent command tests receive the same namespace the server dispatch gets.
    """
    top = argparse.ArgumentParser()
    ConnectParser(top.add_subparsers())
    return top.parse_args(["connect", *argv])
