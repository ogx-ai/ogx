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
def vector_stores():
    """Manage vector stores."""


@click.command("list")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("list vector stores")
def list(ctx):
    """Show available vector stores on distribution endpoint"""

    client = ctx.obj["client"]
    console = Console()
    vector_stores_list_response = client.vector_stores.list()

    if vector_stores_list_response:
        table = Table()
        # Add our specific columns
        table.add_column("id")
        table.add_column("name")
        table.add_column("provider_id")
        table.add_column("provider_vector_store_id")
        table.add_column("params")

        for item in vector_stores_list_response:
            metadata = dict(item.metadata or {})
            provider_id = str(metadata.pop("provider_id", ""))
            provider_vector_store_id = str(
                metadata.pop("provider_vector_store_id", metadata.pop("provider_vector_db_id", ""))
            )
            params = {
                "status": item.status,
                "usage_bytes": item.usage_bytes,
                "embedding_model": metadata.pop("embedding_model", None),
                "embedding_dimension": metadata.pop("embedding_dimension", None),
                "metadata": metadata or None,
            }
            params = yaml.dump(params, default_flow_style=False)

            table.add_row(
                str(item.id),
                str(item.name or ""),
                provider_id,
                provider_vector_store_id,
                params,
            )

        console.print(table)


@vector_stores.command()
@click.help_option("-h", "--help")
@click.argument("vector-db-id")
@click.option("--provider-id", help="Provider ID for the vector db", default=None)
@click.option("--provider-vector-db-id", help="Provider's vector db ID", default=None)
@click.option(
    "--embedding-model",
    type=str,
    help="Embedding model (for vector type)",
    default="all-MiniLM-L6-v2",
)
@click.option(
    "--embedding-dimension",
    type=int,
    help="Embedding dimension (for vector type)",
    default=384,
)
@click.pass_context
@handle_client_errors("register vector store")
def register(
    ctx,
    vector_db_id: str,
    provider_id: str | None,
    provider_vector_db_id: str | None,
    embedding_model: str | None,
    embedding_dimension: int | None,
):
    """Create a new vector store"""
    client = ctx.obj["client"]

    metadata = {"vector_db_id": vector_db_id}
    if provider_id:
        metadata["provider_id"] = provider_id
    if provider_vector_db_id:
        metadata["provider_vector_store_id"] = provider_vector_db_id
    if embedding_model:
        metadata["embedding_model"] = embedding_model
    if embedding_dimension is not None:
        metadata["embedding_dimension"] = embedding_dimension

    response = client.vector_stores.create(name=vector_db_id, metadata=metadata)
    if response:
        click.echo(yaml.dump(response.to_dict()))


@vector_stores.command()
@click.help_option("-h", "--help")
@click.argument("vector-db-id")
@click.pass_context
@handle_client_errors("delete vector store")
def unregister(ctx, vector_db_id: str):
    """Delete a vector store"""
    client = ctx.obj["client"]
    vector_store_id = vector_db_id
    for store in client.vector_stores.list():
        if store.id == vector_db_id or store.name == vector_db_id:
            vector_store_id = store.id
            break
        metadata = store.metadata or {}
        if metadata.get("vector_db_id") == vector_db_id:
            vector_store_id = store.id
            break

    client.vector_stores.delete(vector_store_id=vector_store_id)
    click.echo(f"Vector store '{vector_db_id}' deleted successfully")


# Register subcommands
vector_stores.add_command(list)
vector_stores.add_command(register)
vector_stores.add_command(unregister)
