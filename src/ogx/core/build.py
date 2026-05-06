# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys

from pydantic import BaseModel
from termcolor import cprint

from ogx.core.datatypes import StackConfig
from ogx.core.distribution import get_provider_registry
from ogx.distributions.template import DistributionTemplate
from ogx.log import get_logger
from ogx_api import Api

log = get_logger(name=__name__, category="core")

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


class ApiInput(BaseModel):
    """Input specification pairing an API with its provider type."""

    api: Api
    provider: str


def get_provider_dependencies(
    config: StackConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Get normal and special dependencies from provider configuration."""
    if isinstance(config, DistributionTemplate):
        config = config.build_config()

    deps = []
    external_provider_deps = []
    registry = get_provider_registry(config=config, listing=True)
    for api_str, provider_or_providers in config.providers.items():
        providers_for_api = registry[Api(api_str)]

        if isinstance(provider_or_providers, list):
            providers = provider_or_providers
        else:
            providers = [provider_or_providers]

        for provider in providers:
            # Providers from BuildConfig and RunConfig are subtly different - not great
            provider_type = provider if isinstance(provider, str) else provider.provider_type

            if provider_type not in providers_for_api:
                raise ValueError(f"Provider `{provider}` is not available for API `{api_str}`")

            provider_spec = providers_for_api[provider_type]
            if hasattr(provider_spec, "is_external") and provider_spec.is_external:
                # this ensures we install the top level module for our external providers
                if provider_spec.module:
                    if isinstance(provider_spec.module, str):
                        external_provider_deps.append(provider_spec.module)
                    else:
                        external_provider_deps.extend(provider_spec.module)
            if hasattr(provider_spec, "pip_packages"):
                deps.extend(provider_spec.pip_packages)
            if hasattr(provider_spec, "container_image") and provider_spec.container_image:
                raise ValueError("A stack's dependencies cannot have a container image")

    normal_deps = []
    special_deps = []
    for package in deps:
        if any(f in package for f in ["--no-deps", "--index-url", "--extra-index-url"]):
            special_deps.append(package)
        else:
            normal_deps.append(package)

    return list(set(normal_deps)), list(set(special_deps)), list(set(external_provider_deps))


def print_pip_install_help(config: StackConfig) -> None:
    """Print pip install commands needed for the given stack configuration's provider dependencies.

    Args:
        config: Stack configuration to extract dependencies from.
    """
    normal_deps, special_deps, _ = get_provider_dependencies(config)

    cprint(
        f"Please install needed dependencies using the following commands:\n\nuv pip install {' '.join(normal_deps)}",
        color="yellow",
        file=sys.stderr,
    )
    for special_dep in special_deps:
        cprint(f"uv pip install {special_dep}", color="yellow", file=sys.stderr)
    print()
