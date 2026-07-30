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
@handle_client_errors("list eval tasks")
def list_eval_tasks(ctx):
    """Show available eval tasks on distribution endpoint"""

    client = ctx.obj["client"]
    console = Console()
    headers = []
    eval_tasks_list_response = client.eval_tasks.list()
    if eval_tasks_list_response and len(eval_tasks_list_response) > 0:
        headers = sorted(eval_tasks_list_response[0].__dict__.keys())

    if eval_tasks_list_response:
        table = Table()
        for header in headers:
            table.add_column(header)

        for item in eval_tasks_list_response:
            table.add_row(*[str(getattr(item, header)) for header in headers])
        console.print(table)
