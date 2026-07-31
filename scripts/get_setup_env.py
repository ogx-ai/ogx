#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Small helper script to extract environment variables from a test setup.
Used by integration-tests.sh to set environment variables before starting the server.
"""

import argparse
import sys

from tests.integration.suites import SETUP_DEFINITIONS, SUITE_DEFINITIONS


def get_setup_config(setup_name, suite_name=None):
    """
    Get full configuration (env vars + defaults) for a setup.

    Args:
        setup_name: Name of the setup (e.g., 'ollama', 'gpt')
        suite_name: Optional suite name to get default setup if setup_name is None

    Returns:
        Dictionary with 'env' and 'defaults' keys
    """
    # If no setup specified, try to get default from suite
    if not setup_name and suite_name:
        suite = SUITE_DEFINITIONS.get(suite_name)
        if suite and suite.default_setup:
            setup_name = suite.default_setup

    if not setup_name:
        return {"env": {}, "defaults": {}}

    setup = SETUP_DEFINITIONS.get(setup_name)
    if not setup:
        print(
            f"Error: Unknown setup '{setup_name}'. Available: {', '.join(sorted(SETUP_DEFINITIONS.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    return {"env": setup.env, "defaults": setup.defaults}


def main():
    parser = argparse.ArgumentParser(description="Extract environment variables and defaults from a test setup")
    parser.add_argument("--setup", help="Setup name (e.g., ollama, gpt)")
    parser.add_argument("--suite", help="Suite name to get default setup from if --setup not provided")
    parser.add_argument("--format", choices=["bash", "json"], default="bash", help="Output format (default: bash)")

    args = parser.parse_args()

    config = get_setup_config(args.setup, args.suite)

    if args.format == "bash":
        # Output env vars as bash export statements
        for key, value in config["env"].items():
            print(f"export {key}='{value}'")
        # Output defaults as bash export statements with OGX_TEST_ prefix
        for key, value in config["defaults"].items():
            env_key = f"OGX_TEST_{key.upper()}"
            print(f"export {env_key}='{value}'")
    elif args.format == "json":
        import json

        print(json.dumps(config))


if __name__ == "__main__":
    main()
