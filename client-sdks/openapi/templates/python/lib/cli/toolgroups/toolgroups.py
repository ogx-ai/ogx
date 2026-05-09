# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import click
from rich.console import Console
from rich.table import Table

from ...._types import NOT_GIVEN, NotGiven
from ....types import toolgroup_register_params
from ..common.utils import handle_client_errors


@click.group()
@click.help_option("-h", "--help")
def toolgroups():
    """Manage available tool groups."""


@click.command(name="list", help="Show available llama toolgroups at distribution endpoint")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("list toolgroups")
def list_toolgroups(ctx):
    client = ctx.obj["client"]
    console = Console()

    headers = ["identifier", "provider_id", "args", "mcp_endpoint"]
    response = client.toolgroups.list()
    if response:
        table = Table()
        for header in headers:
            table.add_column(header)

        for item in response:
            row = [str(getattr(item, header)) for header in headers]
            table.add_row(*row)
        console.print(table)


@click.command(name="get")
@click.help_option("-h", "--help")
@click.argument("toolgroup_id")
@click.pass_context
@handle_client_errors("get toolgroup details")
def get_toolgroup(ctx, toolgroup_id: str):
    """Show available llama toolgroups at distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()

    toolgroups_get_response = client.tools.list()
    # filter response to only include provided toolgroup_id
    toolgroups_get_response = [
        toolgroup for toolgroup in toolgroups_get_response if toolgroup.toolgroup_id == toolgroup_id
    ]
    if len(toolgroups_get_response) == 0:
        console.print(
            f"Toolgroup {toolgroup_id} is not found at distribution endpoint. "
            "Please ensure endpoint is serving specified toolgroup.",
            style="bold red",
        )
        return

    headers = sorted(toolgroups_get_response[0].__dict__.keys())
    table = Table()
    for header in headers:
        table.add_column(header)

    for toolgroup in toolgroups_get_response:
        row = [str(getattr(toolgroup, header)) for header in headers]
        table.add_row(*row)
    console.print(table)


@click.command(name="register", help="Register a new toolgroup at distribution endpoint")
@click.help_option("-h", "--help")
@click.argument("toolgroup_id")
@click.option("--provider-id", help="Provider ID for the toolgroup", default=None)
@click.option("--mcp-endpoint", help="JSON mcp_config for the toolgroup", default=None)
@click.option("--args", help="JSON args for the toolgroup", default=None)
@click.pass_context
@handle_client_errors("register toolgroup")
def register_toolgroup(
    ctx,
    toolgroup_id: str,
    provider_id: str | None,
    mcp_endpoint: str | None,
    args: str | None,
):
    """Register a new toolgroup at distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()

    _mcp_endpoint: toolgroup_register_params.McpEndpoint | NotGiven = NOT_GIVEN
    if mcp_endpoint:
        _mcp_endpoint = toolgroup_register_params.McpEndpoint(uri=mcp_endpoint)

    response = client.toolgroups.register(
        toolgroup_id=toolgroup_id,
        provider_id=provider_id,
        args=args,
        mcp_endpoint=_mcp_endpoint,
    )
    if response:
        console.print(f"[green]Successfully registered toolgroup {toolgroup_id}[/green]")


@click.command(name="unregister", help="Unregister a toolgroup from distribution endpoint")
@click.help_option("-h", "--help")
@click.argument("toolgroup_id")
@click.pass_context
@handle_client_errors("unregister toolgroup")
def unregister_toolgroup(ctx, toolgroup_id: str):
    client = ctx.obj["client"]
    console = Console()

    response = client.toolgroups.unregister(toolgroup_id=toolgroup_id)
    if response:
        console.print(f"[green]Successfully deleted toolgroup {toolgroup_id}[/green]")


# Register subcommands
toolgroups.add_command(list_toolgroups)
toolgroups.add_command(get_toolgroup)
toolgroups.add_command(register_toolgroup)
toolgroups.add_command(unregister_toolgroup)
