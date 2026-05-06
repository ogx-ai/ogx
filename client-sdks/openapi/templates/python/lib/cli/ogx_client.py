# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from importlib.metadata import version

import click
import yaml

from ... import LlamaStackClient


def _get_version():
    for dist_name in ("ogx-client", "ogx-open-client"):
        try:
            return version(dist_name)
        except Exception:
            continue
    return "unknown"


from .configure import configure
from .constants import get_config_file_path
from .datasets import datasets
from .eval import eval
from .eval_tasks import eval_tasks
from .inference import inference
from .inspect import inspect
from .models import models
from .post_training import post_training
from .providers import providers
from .scoring_functions import scoring_functions
from .shields import shields
from .toolgroups import toolgroups
from .vector_stores import vector_stores


@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=_get_version(), prog_name="ogx-client")
@click.option("--endpoint", type=str, help="OGX distribution endpoint", default="")
@click.option("--api-key", type=str, help="OGX distribution API key", default="")
@click.option("--config", type=str, help="Path to config file", default=None)
@click.pass_context
def ogx_client(ctx, endpoint: str, api_key: str, config: str | None):
    """Welcome to the ogx-client CLI - a command-line interface for interacting with OGX"""
    ctx.ensure_object(dict)

    # If no config provided, check default location
    if config and endpoint:
        raise ValueError("Cannot use both config and endpoint")

    if config is None:
        default_config = get_config_file_path()
        if default_config.exists():
            config = str(default_config)

    if config:
        try:
            with open(config) as f:
                config_dict = yaml.safe_load(f)
                endpoint = config_dict.get("endpoint", endpoint)
                api_key = config_dict.get("api_key", "")
        except Exception as e:
            click.echo(f"Error loading config from {config}: {str(e)}", err=True)
            click.echo("Falling back to HTTP client with endpoint", err=True)

    if endpoint == "":
        endpoint = "http://localhost:8321"

    default_headers = {}
    if api_key != "":
        default_headers = {
            "Authorization": f"Bearer {api_key}",
        }

    client = LlamaStackClient(
        base_url=endpoint,
        provider_data={
            "fireworks_api_key": os.environ.get("FIREWORKS_API_KEY", ""),
            "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
        default_headers=default_headers,
    )
    ctx.obj = {"client": client}


# Register all subcommands
ogx_client.add_command(models, "models")
ogx_client.add_command(vector_stores, "vector_stores")
ogx_client.add_command(shields, "shields")
ogx_client.add_command(eval_tasks, "eval_tasks")
ogx_client.add_command(providers, "providers")
ogx_client.add_command(datasets, "datasets")
ogx_client.add_command(configure, "configure")
ogx_client.add_command(scoring_functions, "scoring_functions")
ogx_client.add_command(eval, "eval")
ogx_client.add_command(inference, "inference")
ogx_client.add_command(post_training, "post_training")
ogx_client.add_command(inspect, "inspect")
ogx_client.add_command(toolgroups, "toolgroups")


def main():
    ogx_client()


if __name__ == "__main__":
    main()
