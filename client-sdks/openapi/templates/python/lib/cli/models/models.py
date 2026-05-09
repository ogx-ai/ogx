# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import click
from rich.console import Console
from rich.table import Table

from ..common.utils import handle_client_errors


@click.group()
@click.help_option("-h", "--help")
def models():
    """Manage GenAI models."""


@click.command(name="list", help="Show available llama models at distribution endpoint")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("list models")
def list_models(ctx):
    client = ctx.obj["client"]
    console = Console()

    response = client.models.list()
    if response:
        table = Table(
            show_lines=True,  # Add lines between rows for better readability
            padding=(0, 1),  # Add horizontal padding
            expand=True,  # Allow table to use full width
        )

        # Configure columns with specific styling
        table.add_column("model_type", style="blue")
        table.add_column("identifier", style="bold cyan", no_wrap=True, overflow="fold")
        table.add_column("provider_resource_id", style="yellow", no_wrap=True, overflow="fold")
        table.add_column("metadata", style="magenta", max_width=30, overflow="fold")
        table.add_column("provider_id", style="green", max_width=20)

        for item in response:
            table.add_row(
                item.model_type,
                item.identifier,
                item.provider_resource_id,
                str(item.metadata or ""),
                item.provider_id,
            )

        # Create a title for the table
        console.print("\n[bold]Available Models[/bold]\n")
        console.print(table)
        console.print(f"\nTotal models: {len(response)}\n")


@click.command(name="get")
@click.help_option("-h", "--help")
@click.argument("model_id")
@click.pass_context
@handle_client_errors("get model details")
def get_model(ctx, model_id: str):
    """Show details of a specific model at the distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()

    models_get_response = client.models.retrieve(model_id=model_id)

    if not models_get_response:
        console.print(
            f"Model {model_id} is not found at distribution endpoint. "
            "Please ensure endpoint is serving specified model.",
            style="bold red",
        )
        return

    headers = sorted(models_get_response.__dict__.keys())
    table = Table()
    for header in headers:
        table.add_column(header)

    table.add_row(*[str(models_get_response.__dict__[header]) for header in headers])
    console.print(table)


class JSONParamType(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}", param, ctx)


@click.command(name="register", help="Register a new model at distribution endpoint")
@click.help_option("-h", "--help")
@click.argument("model_id")
@click.option("--provider-id", help="Provider ID for the model", default=None)
@click.option("--provider-model-id", help="Provider's model ID", default=None)
@click.option("--metadata", type=JSONParamType(), help="JSON metadata for the model", default=None)
@click.option("--model-type", type=click.Choice(["llm", "embedding"]), default="llm", help="Model type: llm, embedding")
@click.pass_context
@handle_client_errors("register model")
def register_model(
    ctx,
    model_id: str,
    provider_id: str | None,
    provider_model_id: str | None,
    metadata: str | None,
    model_type: str | None,
):
    """Register a new model at distribution endpoint"""
    client = ctx.obj["client"]
    console = Console()

    response = client.models.register(
        model_id=model_id,
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        metadata=metadata,
        model_type=model_type,
    )
    if response:
        console.print(f"[green]Successfully registered model {model_id}[/green]")


@click.command(name="unregister", help="Unregister a model from distribution endpoint")
@click.help_option("-h", "--help")
@click.argument("model_id")
@click.pass_context
@handle_client_errors("unregister model")
def unregister_model(ctx, model_id: str):
    client = ctx.obj["client"]
    console = Console()

    response = client.models.unregister(model_id=model_id)
    if response:
        console.print(f"[green]Successfully deleted model {model_id}[/green]")


# Register subcommands
models.add_command(list_models)
models.add_command(get_model)
models.add_command(register_model)
models.add_command(unregister_model)
