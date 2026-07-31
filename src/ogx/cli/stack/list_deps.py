# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import argparse

from ogx.cli.subcommand import Subcommand


class StackListDeps(Subcommand):
    """CLI subcommand to list pip dependencies for a OGX distribution."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "list-deps",
            prog="ogx list-deps",
            description="list the dependencies for a ogx distribution",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_list_deps_command)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "config",
            type=str,
            nargs="?",  # Make it optional
            metavar="config | distro",
            help="Path to config file to use or name of known distro (ogx list for a list).",
        )

        self.parser.add_argument(
            "--providers",
            type=str,
            default=None,
            help="sync dependencies for a list of providers and only those providers. This list is formatted like: api1=provider1,api2=provider2. Where there can be multiple providers per API.",
        )
        self.parser.add_argument(
            "--format",
            type=str,
            choices=["uv", "deps-only"],
            default="deps-only",
            help="Output format: 'uv' shows shell commands, 'deps-only' shows just the list of dependencies without `uv` (default)",
        )

    def _run_stack_list_deps_command(self, args: argparse.Namespace) -> None:
        # always keep implementation completely silo-ed away from CLI so CLI
        # can be fast to load and reduces dependencies
        if not args.config and not args.providers:
            self.parser.print_help()
            self.parser.exit()

        from ._list_deps import run_stack_list_deps_command

        return run_stack_list_deps_command(args)
