# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from urllib.parse import urlparse

import click
import yaml
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator

from .constants import OGX_CLIENT_CONFIG_DIR, get_config_file_path


def get_config():
    config_file = get_config_file_path()
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return None


@click.command()
@click.help_option("-h", "--help")
@click.option("--endpoint", type=str, help="OGX distribution endpoint", default="")
@click.option("--api-key", type=str, help="OGX distribution API key", default="")
def configure(endpoint: str | None, api_key: str | None):
    """Configure OGX Client CLI."""
    os.makedirs(OGX_CLIENT_CONFIG_DIR, exist_ok=True)
    config_path = get_config_file_path()

    if endpoint != "":
        final_endpoint = endpoint
    else:
        final_endpoint = prompt(
            "> Enter the endpoint of the OGX distribution server: ",
            validator=Validator.from_callable(
                lambda x: len(x) > 0 and (parsed := urlparse(x)).scheme and parsed.netloc,
                error_message="Endpoint cannot be empty and must be a valid URL, please enter a valid endpoint",
            ),
        )

    if api_key != "":
        final_api_key = api_key
    else:
        final_api_key = prompt(
            "> Enter the API key (leave empty if no key is needed): ",
        )

    # Prepare config dict before writing it
    config_dict = {
        "endpoint": final_endpoint,
    }
    if final_api_key != "":
        config_dict["api_key"] = final_api_key

    with open(config_path, "w") as f:
        f.write(
            yaml.dump(
                config_dict,
                sort_keys=True,
            )
        )

    print(f"Done! You can now use the OGX Client CLI with endpoint {final_endpoint}")
