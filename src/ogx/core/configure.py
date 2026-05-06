# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import textwrap
from typing import Any

from ogx.core.datatypes import (
    OGX_RUN_CONFIG_VERSION,
    DistributionSpec,
    Provider,
    StackConfig,
)
from ogx.core.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
)
from ogx.core.stack import cast_distro_name_to_string, replace_env_vars
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.core.utils.prompt_for_config import prompt_for_config
from ogx.log import get_logger
from ogx_api import Api, ProviderSpec

logger = get_logger(name=__name__, category="core")


def configure_single_provider(registry: dict[str, ProviderSpec], provider: Provider) -> Provider:
    """Interactively configure a single provider by prompting for its config values.

    Args:
        registry: Dictionary mapping provider types to their specifications.
        provider: The provider to configure.

    Returns:
        A new Provider instance with the user-provided configuration.
    """
    provider_spec = registry[provider.provider_type]
    config_type = instantiate_class_type(provider_spec.config_class)
    try:
        if provider.config:
            existing = config_type(**provider.config)
        else:
            existing = None
    except Exception:
        existing = None

    cfg = prompt_for_config(config_type, existing)
    return Provider(
        provider_id=provider.provider_id,
        provider_type=provider.provider_type,
        config=cfg.model_dump(),
    )


def configure_api_providers(config: StackConfig, build_spec: DistributionSpec) -> StackConfig:
    """Interactively configure providers for all APIs in the stack configuration.

    Args:
        config: The current stack configuration to update.
        build_spec: The distribution specification defining available providers.

    Returns:
        The updated StackConfig with user-configured providers.
    """
    is_nux = len(config.providers) == 0

    if is_nux:
        logger.info(
            textwrap.dedent(
                """
        OGX is composed of several APIs working together. For each API served by the Stack,
        we need to configure the providers (implementations) you want to use for these APIs.
"""
            )
        )

    provider_registry = get_provider_registry()
    builtin_apis = [a.routing_table_api for a in builtin_automatically_routed_apis()]

    if config.apis:
        apis_to_serve = config.apis
    else:
        apis_to_serve = [a.value for a in Api if a not in (Api.inspect, Api.providers, Api.admin)]

    for api_str in apis_to_serve:
        api = Api(api_str)
        if api in builtin_apis:
            continue
        if api not in provider_registry:
            raise ValueError(f"Unknown API `{api_str}`")

        existing_providers = config.providers.get(api_str, [])
        if existing_providers:
            logger.info("Re-configuring existing providers for API", api=api_str)
            updated_providers = []
            for p in existing_providers:
                logger.info("Configuring provider", provider_type=p.provider_type)
                updated_providers.append(configure_single_provider(provider_registry[api], p))
                logger.info("")
        else:
            # we are newly configuring this API
            plist = build_spec.providers.get(api_str, [])
            plist = plist if isinstance(plist, list) else [plist]

            if not plist:
                raise ValueError(f"No provider configured for API {api_str}?")

            logger.info("Configuring API", api=api_str)
            updated_providers = []
            for i, provider in enumerate(plist):
                if i >= 1:
                    others = ", ".join(p.provider_type for p in plist[i:])
                    logger.info(
                        "Not configuring other providers () interactively. Please edit the resulting YAML directly.\n",
                        others=others,
                    )
                    break

                logger.info("Configuring provider", provider_type=provider.provider_type)
                pid = provider.provider_type.split("::")[-1]
                updated_providers.append(
                    configure_single_provider(
                        provider_registry[api],
                        Provider(
                            provider_id=(f"{pid}-{i:02d}" if len(plist) > 1 else pid),
                            provider_type=provider.provider_type,
                            config={},
                        ),
                    )
                )
                logger.info("")

        config.providers[api_str] = updated_providers

    return config


def upgrade_from_routing_table(
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Upgrade a legacy config dict from routing_table format to the providers format.

    Args:
        config_dict: A configuration dictionary using the old routing_table/api_providers schema.

    Returns:
        The upgraded configuration dictionary using the providers schema.
    """

    def get_providers(entries: Any) -> list[Provider]:
        return [
            Provider(
                provider_id=(f"{entry['provider_type']}-{i:02d}" if len(entries) > 1 else entry["provider_type"]),
                provider_type=entry["provider_type"],
                config=entry["config"],
            )
            for i, entry in enumerate(entries)
        ]

    providers_by_api = {}

    routing_table = config_dict.get("routing_table", {})
    for api_str, entries in routing_table.items():
        providers = get_providers(entries)
        providers_by_api[api_str] = providers

    provider_map = config_dict.get("api_providers", config_dict.get("provider_map", {}))
    if provider_map:
        for api_str, provider in provider_map.items():
            if isinstance(provider, dict) and "provider_type" in provider:
                providers_by_api[api_str] = [
                    Provider(
                        provider_id=f"{provider['provider_type']}",
                        provider_type=provider["provider_type"],
                        config=provider["config"],
                    )
                ]

    config_dict["providers"] = providers_by_api

    config_dict.pop("routing_table", None)
    config_dict.pop("api_providers", None)
    config_dict.pop("provider_map", None)

    config_dict["apis"] = config_dict["apis_to_serve"]
    config_dict.pop("apis_to_serve", None)

    # Add default storage config if not present
    if "storage" not in config_dict:
        config_dict["storage"] = {
            "backends": {
                "kv_default": {
                    "type": "kv_sqlite",
                    "db_path": "~/.ogx/kvstore.db",
                },
                "sql_default": {
                    "type": "sql_sqlite",
                    "db_path": "~/.ogx/sql_store.db",
                },
            },
            "stores": {
                "metadata": {
                    "namespace": "registry",
                    "backend": "kv_default",
                },
                "inference": {
                    "table_name": "inference_store",
                    "backend": "sql_default",
                    "max_write_queue_size": 10000,
                    "num_writers": 4,
                },
                "conversations": {
                    "table_name": "openai_conversations",
                    "backend": "sql_default",
                },
            },
        }

    return config_dict


def parse_and_maybe_upgrade_config(config_dict: dict[str, Any]) -> StackConfig:
    """Parse a configuration dictionary into a StackConfig, upgrading from legacy format if needed.

    Args:
        config_dict: Raw configuration dictionary, potentially in legacy routing_table format.

    Returns:
        A validated StackConfig instance.
    """
    if "routing_table" in config_dict:
        logger.info("Upgrading config...")
        config_dict = upgrade_from_routing_table(config_dict)

    config_dict["version"] = OGX_RUN_CONFIG_VERSION

    processed_config_dict = replace_env_vars(config_dict)
    return StackConfig(**cast_distro_name_to_string(processed_config_dict))
