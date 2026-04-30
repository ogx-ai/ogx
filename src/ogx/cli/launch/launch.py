# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.cli.stack.utils import print_subcommand_description
from ogx.cli.subcommand import Subcommand

from .opencode import LaunchOpenCode


class LaunchParser(Subcommand):
    """Top-level CLI parser for the 'ogx launch' command group."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "launch",
            prog="ogx launch",
            description="Launch third-party tools configured for OGX",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="launch_subcommands")

        LaunchOpenCode.create(subparsers)
        print_subcommand_description(self.parser, subparsers)
