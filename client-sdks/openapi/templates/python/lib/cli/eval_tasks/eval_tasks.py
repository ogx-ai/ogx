# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json

import click
import yaml

from ..common.utils import handle_client_errors
from .list import list_eval_tasks


@click.group()
@click.help_option("-h", "--help")
def eval_tasks():
    """Manage evaluation tasks."""


@eval_tasks.command()
@click.help_option("-h", "--help")
@click.option("--eval-task-id", required=True, help="ID of the eval task")
@click.option("--dataset-id", required=True, help="ID of the dataset to evaluate")
@click.option("--scoring-functions", required=True, multiple=True, help="Scoring functions to use for evaluation")
@click.option("--provider-id", help="Provider ID for the eval task", default=None)
@click.option("--provider-eval-task-id", help="Provider's eval task ID", default=None)
@click.option("--metadata", type=str, help="Metadata for the eval task in JSON format")
@click.pass_context
@handle_client_errors("register eval task")
def register(
    ctx,
    eval_task_id: str,
    dataset_id: str,
    scoring_functions: tuple[str, ...],
    provider_id: str | None,
    provider_eval_task_id: str | None,
    metadata: str | None,
):
    """Register a new eval task"""
    client = ctx.obj["client"]

    if metadata:
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError as err:
            raise click.BadParameter("Metadata must be valid JSON") from err

    response = client.eval_tasks.register(
        eval_task_id=eval_task_id,
        dataset_id=dataset_id,
        scoring_functions=scoring_functions,
        provider_id=provider_id,
        provider_eval_task_id=provider_eval_task_id,
        metadata=metadata,
    )
    if response:
        click.echo(yaml.dump(response.dict()))


# Register subcommands
eval_tasks.add_command(list_eval_tasks)
eval_tasks.add_command(register)
