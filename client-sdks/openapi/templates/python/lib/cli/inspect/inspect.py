# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import click

from .version import inspect_version


@click.group()
@click.help_option("-h", "--help")
def inspect():
    """Inspect server configuration."""


# Register subcommands
inspect.add_command(inspect_version)
