# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.cli.stack.run import add_run_arguments, run_stack_cmd
from ogx.cli.subcommand import Subcommand


class Run(Subcommand):
    """CLI subcommand to start a OGX distribution server."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "run",
            prog="ogx run",
            description="Start the server for a OGX Distribution. You should have already built (or downloaded) and configured the distribution.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_cmd)

    def _add_arguments(self) -> None:
        add_run_arguments(self.parser)

    def _run_cmd(self, args: argparse.Namespace) -> None:
        run_stack_cmd(args, self.parser)
