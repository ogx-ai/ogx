# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import click
from rich.console import Console

from ....models.algorithm_config import AlgorithmConfig
from ....models.training_config import TrainingConfig
from ..common.utils import handle_client_errors


@click.group()
@click.help_option("-h", "--help")
def post_training():
    """Post-training."""


@click.command("supervised_fine_tune")
@click.help_option("-h", "--help")
@click.option("--job-uuid", required=True, help="Job UUID")
@click.option("--model", required=True, help="Model ID")
@click.option("--algorithm-config", required=True, help="Algorithm Config")
@click.option("--training-config", required=True, help="Training Config")
@click.option("--checkpoint-dir", required=False, help="Checkpoint Config", default=None)
@click.pass_context
@handle_client_errors("post_training supervised_fine_tune")
def supervised_fine_tune(
    ctx,
    job_uuid: str,
    model: str,
    algorithm_config: AlgorithmConfig,
    training_config: TrainingConfig,
    checkpoint_dir: str | None,
):
    """Kick off a supervised fine tune job"""
    client = ctx.obj["client"]
    console = Console()

    post_training_job = client.post_training.supervised_fine_tune(
        job_uuid=job_uuid,
        model=model,
        algorithm_config=algorithm_config,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        # logger_config and hyperparam_search_config haven't been used yet
        logger_config={},
        hyperparam_search_config={},
    )
    console.print(post_training_job.job_uuid)


@click.command("list")
@click.help_option("-h", "--help")
@click.pass_context
@handle_client_errors("post_training get_training_jobs")
def get_training_jobs(ctx):
    """Show the list of available post training jobs"""
    client = ctx.obj["client"]
    console = Console()

    post_training_jobs = client.post_training.job.list()
    console.print([post_training_job.job_uuid for post_training_job in post_training_jobs])


@click.command("status")
@click.help_option("-h", "--help")
@click.option("--job-uuid", required=True, help="Job UUID")
@click.pass_context
@handle_client_errors("post_training get_training_job_status")
def get_training_job_status(ctx, job_uuid: str):
    """Show the status of a specific post training job"""
    client = ctx.obj["client"]
    console = Console()

    job_status_reponse = client.post_training.job.status(job_uuid=job_uuid)
    console.print(job_status_reponse)


@click.command("artifacts")
@click.help_option("-h", "--help")
@click.option("--job-uuid", required=True, help="Job UUID")
@click.pass_context
@handle_client_errors("post_training get_training_job_artifacts")
def get_training_job_artifacts(ctx, job_uuid: str):
    """Get the training artifacts of a specific post training job"""
    client = ctx.obj["client"]
    console = Console()

    job_artifacts = client.post_training.job.artifacts(job_uuid=job_uuid)
    console.print(job_artifacts)


@click.command("cancel")
@click.help_option("-h", "--help")
@click.option("--job-uuid", required=True, help="Job UUID")
@click.pass_context
@handle_client_errors("post_training cancel_training_job")
def cancel_training_job(ctx, job_uuid: str):
    """Cancel the training job"""
    client = ctx.obj["client"]

    client.post_training.job.cancel(job_uuid=job_uuid)


# Register subcommands
post_training.add_command(supervised_fine_tune)
post_training.add_command(get_training_jobs)
post_training.add_command(get_training_job_status)
post_training.add_command(get_training_job_artifacts)
post_training.add_command(cancel_training_job)
