# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import click
import yaml
from rich.console import Console

from ..common.utils import handle_client_errors


@click.command(name="inspect")
@click.argument("provider_id")
@click.pass_context
@handle_client_errors("inspect providers")
def inspect_provider(ctx, provider_id):
    """Show specific provider configuration on distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()

    providers_response = client.providers.retrieve(provider_id=provider_id)

    if not providers_response:
        click.secho("Provider not found", fg="red")
        raise click.exceptions.Exit(1)

    console.print(f"provider_id={providers_response.provider_id}")
    console.print(f"provider_type={providers_response.provider_type}")
    console.print("config:")
    for line in yaml.dump(providers_response.config, indent=2).split("\n"):
        console.print(line)
