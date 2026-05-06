# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.log import setup_logging

# Initialize logging early before any loggers get created
setup_logging()

from .stack import StackParser  # type: ignore[attr-defined]
from .stack.utils import print_subcommand_description


class OGXCLIParser:
    """Defines CLI parser for OGX CLI"""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            prog="ogx",
            description="Welcome to the OGX CLI",
            add_help=True,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Default command is to print help
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="subcommands")

        # Add sub-commands
        StackParser.create(subparsers)

        print_subcommand_description(self.parser, subparsers)

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        if not isinstance(args, argparse.Namespace):
            raise TypeError(f"Expected argparse.Namespace, got {type(args)}")
        return args

    def run(self, args: argparse.Namespace) -> None:
        args.func(args)


def main() -> None:
    """Entry point for the OGX CLI."""
    parser = OGXCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
