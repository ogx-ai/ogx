# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import os
import signal
import subprocess
import sys

from termcolor import cprint

from ogx.log import get_logger

log = get_logger(name=__name__, category="core")


def formulate_run_args(image_type: str, distro_name: str) -> list:
    """Build the command-line arguments for starting a OGX server.

    Args:
        image_type: The image type (e.g., 'venv', 'container').
        distro_name: The distribution or virtual environment name.

    Returns:
        A list of command-line arguments, or an empty list if no environment is detected.
    """
    # Only venv is supported now
    current_venv = os.environ.get("VIRTUAL_ENV")
    env_name = distro_name or current_venv
    if not env_name:
        cprint(
            "No current virtual environment detected, please specify a virtual environment name with --image-name",
            color="red",
            file=sys.stderr,
        )
        return []

    cprint(f"Using virtual environment: {env_name}", file=sys.stderr)

    script = importlib.resources.files("ogx") / "core/start_stack.sh"
    run_args = [
        script,
        image_type,
        env_name,
    ]

    return run_args


def in_notebook():
    """Detect whether the current code is running inside a Jupyter notebook.

    Returns:
        True if running in a Jupyter/IPython notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def run_command(command: list[str]) -> int:
    """
    Run a command with interrupt handling and output capture.
    Uses subprocess.run with direct stream piping for better performance.

    Args:
        command (list): The command to run.

    Returns:
        int: The return code of the command.
    """
    original_sigint = signal.getsignal(signal.SIGINT)
    ctrl_c_pressed = False

    def sigint_handler(signum, frame):
        nonlocal ctrl_c_pressed
        ctrl_c_pressed = True
        log.info("\nCtrl-C detected. Aborting...")

    try:
        # Set up the signal handler
        signal.signal(signal.SIGINT, sigint_handler)

        # Run the command with stdout/stderr piped directly to system streams
        result = subprocess.run(
            command,
            text=True,
            check=False,
        )

        # Print stdout and stderr if command failed
        if result.returncode != 0:
            log.error(f"Command {' '.join(command)} failed with returncode {result.returncode}")
            if result.stdout:
                log.error(f"STDOUT: {result.stdout}")
            if result.stderr:
                log.error(f"STDERR: {result.stderr}")

        return result.returncode
    except subprocess.SubprocessError as e:
        log.error(f"Subprocess error: {e}")
        return 1
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return 1
    finally:
        # Restore the original signal handler
        signal.signal(signal.SIGINT, original_sigint)
