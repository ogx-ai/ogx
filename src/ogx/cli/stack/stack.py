# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from importlib.metadata import version

from ogx.cli.stack.list_stacks import StackListBuilds
from ogx.cli.stack.utils import print_subcommand_description
from ogx.cli.subcommand import Subcommand

from .lets_go import StackLetsGo
from .list_apis import StackListApis
from .list_deps import StackListDeps
from .list_providers import StackListProviders
from .remove import StackRemove
from .run import StackRun


class StackParser(Subcommand):
    """Top-level CLI parser for the 'ogx' command group."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "stack",
            prog="ogx",
            description="Operations for the OGX / Distributions",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        self.parser.add_argument(
            "--version",
            action="version",
            version=f"{version('ogx')}",
        )

        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="stack_subcommands")

        # Add sub-commands
        StackListDeps.create(subparsers)
        StackListApis.create(subparsers)
        StackListProviders.create(subparsers)
        StackRun.create(subparsers)
        StackLetsGo.create(subparsers)
        StackRemove.create(subparsers)
        StackListBuilds.create(subparsers)
        print_subcommand_description(self.parser, subparsers)
