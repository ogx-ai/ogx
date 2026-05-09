# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import sys
from pathlib import Path

import yaml
from termcolor import cprint

from ogx.core.build import get_provider_dependencies
from ogx.core.datatypes import StackConfig
from ogx.core.distribution import get_provider_registry
from ogx.core.stack import run_config_from_dynamic_config_spec
from ogx.log import get_logger

from .utils import add_dependent_providers

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "templates"

logger = get_logger(name=__name__, category="cli")


# These are the dependencies needed by the distribution server.
# `ogx` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "aiosqlite",
    "fastapi",
    "fire",
    "httpx",
    "uvicorn",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
]


def format_output_deps_only(
    normal_deps: list[str],
    special_deps: list[str],
    external_deps: list[str],
    uv: bool = False,
) -> str:
    """Format dependencies as a list."""
    lines = []

    uv_str = ""
    if uv:
        uv_str = "uv pip install "
        # Only quote when emitting a shell command. In deps-only mode, keep raw
        # specs so they can be safely consumed via command substitution.
        formatted_normal_deps = [quote_if_needed(dep) for dep in normal_deps]
    else:
        formatted_normal_deps = normal_deps
    lines.append(f"{uv_str}{' '.join(formatted_normal_deps)}")

    for special_dep in special_deps:
        formatted = quote_special_dep(special_dep) if uv else special_dep
        lines.append(f"{uv_str}{formatted}")

    for external_dep in external_deps:
        formatted = quote_special_dep(external_dep) if uv else external_dep
        lines.append(f"{uv_str}{formatted}")

    return "\n".join(lines)


def run_stack_list_deps_command(args: argparse.Namespace) -> None:
    """Resolve and print the pip dependencies for a OGX distribution.

    Args:
        args: parsed CLI arguments containing config or providers specification.
    """
    if args.config:
        try:
            from ogx.core.utils.config_resolution import resolve_config_or_distro

            config_file = resolve_config_or_distro(args.config)
        except ValueError as e:
            cprint(
                f"Could not parse config file {args.config}: {e}",
                color="red",
                file=sys.stderr,
            )
            sys.exit(1)
        if config_file:
            with open(config_file) as f:
                try:
                    contents = yaml.safe_load(f)
                    # Remove auth provider_config to avoid validation errors with env var syntax.
                    # We only need provider dependencies, not auth config (auth has no pip_packages).
                    # This is simpler than modifying the schema to accept type="" which would require
                    # removing discriminated union and adding custom validation logic and modifying
                    # all 4 auth provider config classes (a very invasive change)
                    if "server" in contents and "auth" in contents["server"]:
                        if "provider_config" in contents["server"]["auth"]:
                            contents["server"]["auth"]["provider_config"] = None
                    config = StackConfig(**contents)
                except Exception as e:
                    cprint(
                        f"Could not parse config file {config_file}: {e}",
                        color="red",
                        file=sys.stderr,
                    )
                    sys.exit(1)
    elif args.providers:
        try:
            config = run_config_from_dynamic_config_spec(args.providers)
        except ValueError as e:
            cprint(str(e), color="red", file=sys.stderr)
            sys.exit(1)
        # Expand dependent providers (e.g. agents depends on inference, safety, etc.)
        provider_registry = get_provider_registry()
        requested_provider_types = list(
            {provider.provider_type for providers in config.providers.values() for provider in providers}
        )
        add_dependent_providers(
            provider_list=config.providers,
            provider_registry=provider_registry,
            requested_provider_types=requested_provider_types,
        )

    normal_deps, special_deps, external_provider_dependencies = get_provider_dependencies(config)
    normal_deps += SERVER_DEPENDENCIES

    # Add external API dependencies
    if config.external_apis_dir:
        from ogx.core.external import load_external_apis

        external_apis = load_external_apis(config)
        if external_apis:
            for _, api_spec in external_apis.items():
                normal_deps.extend(api_spec.pip_packages)

    # Format and output based on requested format
    output = format_output_deps_only(
        normal_deps=normal_deps,
        special_deps=special_deps,
        external_deps=external_provider_dependencies,
        uv=args.format == "uv",
    )

    print(output)


def quote_if_needed(dep: str) -> str:
    """Wrap a dependency string in quotes if it contains shell-special characters.

    Args:
        dep: a pip dependency specifier string.

    Returns:
        The dependency string, quoted if it contains commas or comparison operators.
    """
    # Add quotes if the dependency contains special characters that need escaping in shell
    # This includes: commas, comparison operators (<, >, <=, >=, ==, !=)
    needs_quoting = any(char in dep for char in [",", "<", ">", "="])
    return f"'{dep}'" if needs_quoting else dep


def quote_special_dep(dep_string: str) -> str:
    """
    Quote individual packages in a special dependency string.
    Special deps may contain multiple packages and flags like --extra-index-url.
    We need to quote only the package specs that contain special characters.
    """
    parts = dep_string.split()
    quoted_parts = []

    for part in parts:
        # Don't quote flags (they start with -)
        if part.startswith("-"):
            quoted_parts.append(part)
        else:
            # Quote package specs that need it
            quoted_parts.append(quote_if_needed(part))

    return " ".join(quoted_parts)
