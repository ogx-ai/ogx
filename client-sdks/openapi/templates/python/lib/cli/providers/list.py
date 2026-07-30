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
@handle_client_errors("list providers")
def list_providers(ctx):
    """Show available providers on distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()
    headers = ["API", "Provider ID", "Provider Type"]

    providers_response = client.providers.list()
    table = Table()
    for header in headers:
        table.add_column(header)

    for response in providers_response:
        table.add_row(response.api, response.provider_id, response.provider_type)

    console.print(table)
