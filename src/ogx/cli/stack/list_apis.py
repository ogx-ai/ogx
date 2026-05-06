# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.cli.subcommand import Subcommand


class StackListApis(Subcommand):
    """CLI subcommand to list all APIs in the OGX implementation."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "list-apis",
            prog="ogx list-apis",
            description="List APIs part of the OGX implementation",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_apis_list_cmd)

    def _add_arguments(self) -> None:
        pass

    def _run_apis_list_cmd(self, args: argparse.Namespace) -> None:
        from ogx.cli.table import print_table
        from ogx.core.distribution import stack_apis

        # eventually, this should query a registry at llama.meta.com/ogx/distributions
        headers = [
            "API",
        ]

        rows = []
        for api in stack_apis():
            rows.append(
                [
                    api.value,
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
