# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import click
import yaml

from .list import list_scoring_functions


@click.group()
@click.help_option("-h", "--help")
def scoring_functions():
    """Manage scoring functions."""


@scoring_functions.command()
@click.help_option("-h", "--help")
@click.option("--scoring-fn-id", required=True, help="Id of the scoring function")
@click.option("--description", required=True, help="Description of the scoring function")
@click.option("--return-type", type=str, required=True, help="Return type of the scoring function")
@click.option("--provider-id", type=str, help="Provider ID for the scoring function", default=None)
@click.option("--provider-scoring-fn-id", type=str, help="Provider's scoring function ID", default=None)
@click.option("--params", type=str, help="Parameters for the scoring function in JSON format", default=None)
@click.pass_context
def register(
    ctx,
    scoring_fn_id: str,
    description: str,
    return_type: str,
    provider_id: str | None,
    provider_scoring_fn_id: str | None,
    params: str | None,
):
    """Register a new scoring function"""
    client = ctx.obj["client"]

    if params:
        try:
            params = json.loads(params)
        except json.JSONDecodeError as err:
            raise click.BadParameter("Parameters must be valid JSON") from err

    response = client.scoring_functions.register(
        scoring_fn_id=scoring_fn_id,
        description=description,
        return_type=json.loads(return_type),
        provider_id=provider_id,
        provider_scoring_fn_id=provider_scoring_fn_id,
        params=params,
    )
    if response:
        click.echo(yaml.dump(response.dict()))


# Register subcommands
scoring_functions.add_command(list_scoring_functions)
scoring_functions.add_command(register)
