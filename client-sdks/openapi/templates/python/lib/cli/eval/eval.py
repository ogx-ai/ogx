# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import click

from .run_benchmark import run_benchmark
from .run_scoring import run_scoring


@click.group()
@click.help_option("-h", "--help")
def eval():
    """Run evaluation tasks."""


# Register subcommands
eval.add_command(run_benchmark)
eval.add_command(run_scoring)
