# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from ogx.core.datatypes import (
    OGX_RUN_CONFIG_VERSION,
    Provider,
    StackConfig,
)
from ogx.core.stack import cast_distro_name_to_string, replace_env_vars
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.core.utils.prompt_for_config import prompt_for_config
from ogx.log import get_logger
from ogx_api import ProviderSpec

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


def _migrate_prompts_kv_to_sql(config_dict: dict[str, Any]) -> None:
    """Migrate prompts store from legacy KVStoreReference to SqlStoreReference in-place."""
    prompts_cfg = config_dict.get("storage", {}).get("stores", {}).get("prompts")
    if prompts_cfg and "namespace" in prompts_cfg and "table_name" not in prompts_cfg:
        logger.info("Migrating prompts store config from KVStoreReference to SqlStoreReference")
        prompts_cfg["table_name"] = prompts_cfg.pop("namespace")
        backend = prompts_cfg.get("backend", "")
        if backend == "kv_default":
            prompts_cfg["backend"] = "sql_default"
        elif backend.startswith("kv_"):
            sql_backend = "sql_" + backend[3:]
            logger.info(
                "Remapping prompts backend from KV to SQL variant",
                kv_backend=backend,
                sql_backend=sql_backend,
            )
            prompts_cfg["backend"] = sql_backend


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

    _migrate_prompts_kv_to_sql(config_dict)

    config_dict["version"] = OGX_RUN_CONFIG_VERSION

    processed_config_dict = replace_env_vars(config_dict)
    return StackConfig(**cast_distro_name_to_string(processed_config_dict))
