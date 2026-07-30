# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from ogx.cli.subcommand import Subcommand


class StackListProviders(Subcommand):
    """CLI subcommand to list available providers for OGX APIs."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "list-providers",
            prog="ogx list-providers",
            description="Show available OGX Providers for an API",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_providers_list_cmd)

    @property
    def providable_apis(self) -> list[str]:
        from ogx.core.distribution import providable_apis

        return [api.value for api in providable_apis()]

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "api",
            type=str,
            choices=self.providable_apis,
            nargs="?",
            help="API to list providers for. List all if not specified.",
        )

    def _run_providers_list_cmd(self, args: argparse.Namespace) -> None:
        from ogx.cli.table import print_table
        from ogx.core.distribution import Api, get_provider_registry

        all_providers = get_provider_registry()
        if args.api:
            providers = [(args.api, all_providers[Api(args.api)])]
        else:
            providers = [(k.value, prov) for k, prov in all_providers.items()]

        providers = [(api, p) for api, p in providers if api in self.providable_apis]

        # eventually, this should query a registry at llama.meta.com/ogx/distributions
        headers = [
            "API Type",
            "Provider Type",
            "PIP Package Dependencies",
        ]

        rows = []

        specs = [spec for api, p in providers for spec in p.values()]
        for spec in specs:
            if spec.is_sample:
                continue
            rows.append(
                [
                    spec.api.value,
                    spec.provider_type,
                    ",".join(spec.pip_packages) if hasattr(spec, "pip_packages") else "",
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
            sort_by=(0, 1),
        )
