# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path
from typing import Any

from ogx.core.datatypes import Provider, ProviderSpec
from ogx.core.utils.dynamic import instantiate_class_type
from ogx_api import Api

TEMPLATES_PATH = Path(__file__).parent.parent.parent / "distributions"


class ImageType(Enum):
    """Supported image types for building OGX distributions."""

    CONTAINER = "container"
    VENV = "venv"


def print_subcommand_description(parser: argparse.ArgumentParser, subparsers: argparse._SubParsersAction[Any]) -> None:
    """Print descriptions of subcommands."""
    description_text = ""
    for name, subcommand in subparsers.choices.items():
        description = subcommand.description
        description_text += f"  {name:<21} {description}\n"
    parser.epilog = description_text


def add_dependent_providers(
    provider_list: dict[str, list[Provider]],
    provider_registry: dict[Api, dict[str, ProviderSpec]],
    requested_provider_types: list[str],
    *,
    distro_dir: str | None = None,
    include_configs: bool = False,
) -> None:
    def add_provider_for_api(api: Api) -> None:
        api_key = api.value
        if api_key in provider_list and provider_list[api_key]:
            return
        providers_for_api = provider_registry.get(api)
        if not providers_for_api:
            return
        provider_spec = next(
            (spec for spec in providers_for_api.values() if spec.provider_type.startswith("inline::")),
            None,
        )
        if provider_spec is None:
            provider_spec = next(iter(providers_for_api.values()), None)
        if provider_spec is None:
            return

        if include_configs:
            if not distro_dir:
                raise ValueError("distro_dir is required when include_configs=True")
            config_type = instantiate_class_type(provider_spec.config_class)
            if config_type is not None and hasattr(config_type, "sample_run_config"):
                config = config_type.sample_run_config(__distro_dir__=distro_dir)
            else:
                config = {}
            provider = Provider(
                provider_type=provider_spec.provider_type,
                config=config,
                provider_id=provider_spec.provider_type.split("::")[1],
            )
        else:
            provider = Provider(
                provider_type=provider_spec.provider_type,
                provider_id=provider_spec.provider_type.split("::")[1],
                module=None,
            )
        provider_list.setdefault(api_key, []).append(provider)

    def expand_dependencies(provider_spec: ProviderSpec) -> None:
        for api in provider_spec.api_dependencies:
            add_provider_for_api(api)
            for candidate in provider_list.get(api.value, []):
                candidate_spec = provider_registry[api].get(candidate.provider_type)
                if candidate_spec:
                    expand_dependencies(candidate_spec)

    for provider_type in requested_provider_types:
        for _, api_providers in provider_registry.items():
            provider_spec = api_providers.get(provider_type)
            if provider_spec:
                expand_dependencies(provider_spec)
