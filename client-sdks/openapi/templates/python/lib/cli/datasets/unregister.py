# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import click

from ..common.utils import handle_client_errors


@click.command("unregister")
@click.help_option("-h", "--help")
@click.argument("dataset-id", required=True)
@click.pass_context
@handle_client_errors("unregister dataset")
def unregister(ctx, dataset_id: str):
    """Remove a dataset"""
    client = ctx.obj["client"]
    client.datasets.unregister(dataset_id=dataset_id)
    click.echo(f"Dataset '{dataset_id}' unregistered successfully")
