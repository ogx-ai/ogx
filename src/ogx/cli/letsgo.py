# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from typing import Any

from ogx.cli.stack.lets_go import add_letsgo_arguments, run_letsgo_cmd
from ogx.cli.subcommand import Subcommand


class LetsGo(Subcommand):
    """Auto-detect providers, generate runtime config, and start the stack."""

    def __init__(self, subparsers: Any) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "letsgo",
            prog="ogx letsgo",
            description="Auto-detect providers and start the stack",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_cmd)

    def _add_arguments(self) -> None:
        add_letsgo_arguments(self.parser)

    def _run_cmd(self, args: argparse.Namespace) -> None:
        run_letsgo_cmd(args, self.parser)
