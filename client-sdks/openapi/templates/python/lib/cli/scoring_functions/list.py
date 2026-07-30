# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import click
from rich.console import Console
from rich.table import Table

from ..common.utils import handle_client_errors


@click.command("list")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("list scoring functions")
def list_scoring_functions(ctx):
    """Show available scoring functions on distribution endpoint"""

    client = ctx.obj["client"]
    console = Console()
    headers = [
        "identifier",
        "provider_id",
        "description",
        "type",
    ]

    scoring_functions_list_response = client.scoring_functions.list()
    if scoring_functions_list_response:
        table = Table()
        for header in headers:
            table.add_column(header)

        for item in scoring_functions_list_response:
            table.add_row(*[str(getattr(item, header)) for header in headers])
        console.print(table)
