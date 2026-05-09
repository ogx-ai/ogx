# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import click

from .list import list_datasets
from .register import register
from .unregister import unregister


@click.group()
@click.help_option("-h", "--help")
def datasets():
    """Manage datasets."""


# Register subcommands
datasets.add_command(list_datasets)
datasets.add_command(register)
datasets.add_command(unregister)
