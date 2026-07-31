# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import click
from rich.console import Console

from ..common.utils import handle_client_errors


@click.command("version")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("inspect version")
def inspect_version(ctx):
    """Show OGX version on distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()
    version_response = client.inspect.version()
    console.print(version_response)
