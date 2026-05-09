# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import click
import yaml
from rich.console import Console
from rich.table import Table

from ..common.utils import handle_client_errors


@click.group()
@click.help_option("-h", "--help")
def shields():
    """Manage safety shield services."""


@click.command("list")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("list shields")
def list(ctx):
    """Show available safety shields on distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()

    shields_list_response = client.shields.list()

    if shields_list_response:
        table = Table(
            show_lines=True,  # Add lines between rows for better readability
            padding=(0, 1),  # Add horizontal padding
            expand=True,  # Allow table to use full width
        )

        table.add_column("identifier", style="bold cyan", no_wrap=True, overflow="fold")
        table.add_column("provider_alias", style="yellow", no_wrap=True, overflow="fold")
        table.add_column("params", style="magenta", max_width=30, overflow="fold")
        table.add_column("provider_id", style="green", max_width=20)

        for item in shields_list_response:
            table.add_row(
                item.identifier,
                item.provider_resource_id,
                str(item.params or ""),
                item.provider_id,
            )

        console.print(table)


@shields.command()
@click.help_option("-h", "--help")
@click.option("--shield-id", required=True, help="Id of the shield")
@click.option("--provider-id", help="Provider ID for the shield", default=None)
@click.option("--provider-shield-id", help="Provider's shield ID", default=None)
@click.option(
    "--params",
    type=str,
    help="JSON configuration parameters for the shield",
    default=None,
)
@click.pass_context
@handle_client_errors("register shield")
def register(
    ctx,
    shield_id: str,
    provider_id: str | None,
    provider_shield_id: str | None,
    params: str | None,
):
    """Register a new safety shield"""
    client = ctx.obj["client"]

    response = client.shields.register(
        shield_id=shield_id,
        params=params,
        provider_id=provider_id,
        provider_shield_id=provider_shield_id,
    )
    if response:
        click.echo(yaml.dump(response.dict()))


# Register subcommands
shields.add_command(list)
shields.add_command(register)
