# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import shutil
import sys
from pathlib import Path

from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.cli.table import print_table


class StackRemove(Subcommand):
    """Remove the build stack"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "rm",
            prog="ogx rm",
            description="Remove the build stack",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._remove_stack_build_command)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "name",
            type=str,
            nargs="?",
            help="Name of the stack to delete",
        )
        self.parser.add_argument(
            "--all",
            "-a",
            action="store_true",
            help="Delete all stacks (use with caution)",
        )

    def _get_distribution_dirs(self) -> dict[str, Path]:
        """Return a dictionary of distribution names and their paths"""
        distributions = {}
        dist_dir = Path.home() / ".llama" / "distributions"

        if dist_dir.exists():
            for stack_dir in dist_dir.iterdir():
                if stack_dir.is_dir():
                    distributions[stack_dir.name] = stack_dir
        return distributions

    def _list_stacks(self) -> None:
        """Display available stacks in a table"""
        distributions = self._get_distribution_dirs()
        if not distributions:
            cprint("No stacks found in ~/.llama/distributions", color="red", file=sys.stderr)
            sys.exit(1)

        headers = ["Stack Name", "Path"]
        rows = [[name, str(path)] for name, path in distributions.items()]
        print_table(rows, headers, separate_rows=True)

    def _remove_stack_build_command(self, args: argparse.Namespace) -> None:
        distributions = self._get_distribution_dirs()

        if args.all:
            confirm = input("Are you sure you want to delete ALL stacks? [yes-i-really-want/N] ").lower()
            if confirm != "yes-i-really-want":
                cprint("Deletion cancelled.", color="green", file=sys.stderr)
                return

            for name, path in distributions.items():
                try:
                    shutil.rmtree(path)
                    cprint(f"Deleted stack: {name}", color="green", file=sys.stderr)
                except Exception as e:
                    cprint(
                        f"Failed to delete stack {name}: {e}",
                        color="red",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        if not args.name:
            self._list_stacks()
            if not args.name:
                return

        if args.name not in distributions:
            self._list_stacks()
            cprint(
                f"Stack not found: {args.name}",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)

        stack_path = distributions[args.name]

        confirm = input(f"Are you sure you want to delete stack '{args.name}'? [y/N] ").lower()
        if confirm != "y":
            cprint("Deletion cancelled.", color="green", file=sys.stderr)
            return

        try:
            shutil.rmtree(stack_path)
            cprint(f"Successfully deleted stack: {args.name}", color="green", file=sys.stderr)
        except Exception as e:
            cprint(f"Failed to delete stack {args.name}: {e}", color="red", file=sys.stderr)
            sys.exit(1)
