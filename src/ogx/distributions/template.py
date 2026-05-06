# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
from typing import Any, Literal

import jinja2
import rich
import yaml
from pydantic import BaseModel, Field

from ogx.core.datatypes import (
    OGX_RUN_CONFIG_VERSION,
    Api,
    BuildProvider,
    ModelInput,
    Provider,
    SafetyConfig,
    ShieldInput,
    VectorStoresConfig,
)
from ogx.core.distribution import get_provider_registry
from ogx.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    SqlStoreReference,
    StorageBackendType,
)
from ogx.core.storage.kvstore.config import SqliteKVStoreConfig
from ogx.core.storage.sqlstore.sqlstore import SqliteSqlStoreConfig
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.providers.utils.inference.model_registry import ProviderModelEntry
from ogx_api import ConnectorInput, ConnectorType, ModelType


def filter_empty_values(obj: Any) -> Any:
    """Recursively filter out specific empty values from a dictionary or list.

    This function removes:
    - Empty strings ('') only when they are the 'module' field
    - Empty dictionaries ({}) only when they are the 'config' field
    - None values (always excluded)
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        filtered = {}
        for key, value in obj.items():
            # Special handling for specific fields
            if key == "module" and isinstance(value, str) and value == "":
                # Skip empty module strings
                continue
            elif key == "config" and isinstance(value, dict) and not value:
                # Skip empty config dictionaries
                continue
            elif key == "container_image" and not value:
                # Skip empty container_image names
                continue
            else:
                # For all other fields, recursively filter but preserve empty values
                filtered_value = filter_empty_values(value)
                # if filtered_value is not None:
                filtered[key] = filtered_value
        return filtered

    elif isinstance(obj, list):
        filtered = []
        for item in obj:
            filtered_item = filter_empty_values(item)
            if filtered_item is not None:
                filtered.append(filtered_item)
        return filtered

    else:
        # For all other types (including empty strings and dicts that aren't module/config),
        # preserve them as-is
        return obj


def get_model_registry(
    available_models: dict[str, list[ProviderModelEntry]],
) -> tuple[list[ModelInput], bool]:
    """Build a model registry from provider model entries, detecting ID conflicts.

    Args:
        available_models: mapping of provider IDs to their available model entries.

    Returns:
        A tuple of (list of ModelInput, whether model ID conflicts were detected).
    """
    models = []

    # check for conflicts in model ids
    all_ids = set()
    ids_conflict = False

    for _, entries in available_models.items():
        for entry in entries:
            ids = [entry.provider_model_id] + entry.aliases
            for model_id in ids:
                if model_id in all_ids:
                    ids_conflict = True
                    rich.print(
                        f"[yellow]Model id {model_id} conflicts; all model ids will be prefixed with provider id[/yellow]"
                    )
                    break
            all_ids.update(ids)
            if ids_conflict:
                break
        if ids_conflict:
            break

    for provider_id, entries in available_models.items():
        for entry in entries:
            ids = [entry.provider_model_id] + entry.aliases
            for model_id in ids:
                identifier = f"{provider_id}/{model_id}" if ids_conflict and provider_id not in model_id else model_id
                models.append(
                    ModelInput(
                        model_id=identifier,
                        provider_model_id=entry.provider_model_id,
                        provider_id=provider_id,
                        model_type=entry.model_type,
                        metadata=entry.metadata,
                    )
                )
    return models, ids_conflict


def get_shield_registry(
    available_safety_models: dict[str, list[ProviderModelEntry]],
    ids_conflict_in_models: bool,
) -> list[ShieldInput]:
    """Build a shield registry from safety model entries, detecting ID conflicts.

    Args:
        available_safety_models: mapping of provider IDs to their safety model entries.
        ids_conflict_in_models: whether model ID conflicts were detected in the model registry.

    Returns:
        A list of ShieldInput instances for registered shields.
    """
    shields = []

    # check for conflicts in shield ids
    all_ids = set()
    ids_conflict = False

    for _, entries in available_safety_models.items():
        for entry in entries:
            ids = [entry.provider_model_id] + entry.aliases
            for model_id in ids:
                if model_id in all_ids:
                    ids_conflict = True
                    rich.print(
                        f"[yellow]Shield id {model_id} conflicts; all shield ids will be prefixed with provider id[/yellow]"
                    )
                    break
            all_ids.update(ids)
            if ids_conflict:
                break
        if ids_conflict:
            break

    for provider_id, entries in available_safety_models.items():
        for entry in entries:
            ids = [entry.provider_model_id] + entry.aliases
            for model_id in ids:
                identifier = f"{provider_id}/{model_id}" if ids_conflict and provider_id not in model_id else model_id
                shields.append(
                    ShieldInput(
                        shield_id=identifier,
                        provider_shield_id=f"{provider_id}/{entry.provider_model_id}"
                        if ids_conflict_in_models
                        else entry.provider_model_id,
                    )
                )

    return shields


class DefaultModel(BaseModel):
    """A model entry used for documentation generation in distribution templates."""

    model_id: str
    doc_string: str


class RunConfigSettings(BaseModel):
    """Settings for generating a distribution run configuration YAML file."""

    provider_overrides: dict[str, list[Provider]] = Field(default_factory=dict)
    default_models: list[ModelInput] | None = None
    default_shields: list[ShieldInput] | None = None
    default_connectors: list[ConnectorInput] | None = None
    vector_stores_config: VectorStoresConfig | None = None
    safety_config: SafetyConfig | None = None
    auth_config: dict[str, Any] | None = None
    storage_backends: dict[str, Any] | None = None
    storage_stores: dict[str, Any] | None = None

    def run_config(
        self,
        name: str,
        providers: dict[str, list[BuildProvider]],
        container_image: str | None = None,
    ) -> dict:
        provider_registry = get_provider_registry()
        provider_configs = {}
        for api_str, provider_objs in providers.items():
            if api_providers := self.provider_overrides.get(api_str):
                # Convert Provider objects to dicts for YAML serialization
                provider_configs[api_str] = [p.model_dump(exclude_none=True) for p in api_providers]
                continue

            provider_configs[api_str] = []
            for provider in provider_objs:
                api = Api(api_str)
                if provider.provider_type not in provider_registry[api]:
                    raise ValueError(f"Unknown provider type: {provider.provider_type} for API: {api_str}")
                provider_id = provider.provider_type.split("::")[-1]
                config_class = provider_registry[api][provider.provider_type].config_class
                assert config_class is not None, (
                    f"No config class for provider type: {provider.provider_type} for API: {api_str}"
                )

                config_class = instantiate_class_type(config_class)
                if hasattr(config_class, "sample_run_config"):
                    config = config_class.sample_run_config(__distro_dir__=f"~/.llama/distributions/{name}")
                else:
                    config = {}
                # BuildProvider does not have a config attribute; skip assignment
                provider_configs[api_str].append(
                    Provider(
                        provider_id=provider_id,
                        provider_type=provider.provider_type,
                        config=config,
                    ).model_dump(exclude_none=True)
                )
        # Get unique set of APIs from providers
        apis = sorted(providers.keys())

        storage_backends = self.storage_backends or {
            "kv_default": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=f"~/.llama/distributions/{name}",
                db_name="kvstore.db",
            ),
            "sql_default": SqliteSqlStoreConfig.sample_run_config(
                __distro_dir__=f"~/.llama/distributions/{name}",
                db_name="sql_store.db",
            ),
        }

        storage_stores = self.storage_stores or {
            "metadata": KVStoreReference(
                backend="kv_default",
                namespace="registry",
            ).model_dump(exclude_none=True),
            "inference": InferenceStoreReference(
                backend="sql_default",
                table_name="inference_store",
            ).model_dump(exclude_none=True),
            "conversations": SqlStoreReference(
                backend="sql_default",
                table_name="openai_conversations",
            ).model_dump(exclude_none=True),
            "prompts": KVStoreReference(
                backend="kv_default",
                namespace="prompts",
            ).model_dump(exclude_none=True),
            "connectors": KVStoreReference(
                backend="kv_default",
                namespace="connectors",
            ).model_dump(exclude_none=True),
        }

        storage_config = dict(
            backends=storage_backends,
            stores=storage_stores,
        )

        # Return a dict that matches StackRunConfig structure
        config = {
            "version": OGX_RUN_CONFIG_VERSION,
            "distro_name": name,
            "container_image": container_image,
            "apis": apis,
            "providers": provider_configs,
            "storage": storage_config,
            "registered_resources": {
                "models": [m.model_dump(exclude_none=True) for m in (self.default_models or [])],
                "shields": [s.model_dump(exclude_none=True) for s in (self.default_shields or [])],
                "vector_dbs": [],
            },
            "server": {
                "port": 8321,
            },
        }

        if self.auth_config:
            config["server"]["auth"] = self.auth_config

        if self.vector_stores_config:
            config["vector_stores"] = self.vector_stores_config.model_dump(exclude_none=True)

        if self.safety_config:
            config["safety"] = self.safety_config.model_dump(exclude_none=True)

        if self.default_connectors is not None:
            config["connectors"] = [c.model_dump(exclude_none=True) for c in self.default_connectors]

        return config


class DistributionTemplate(BaseModel):
    """
    Represents a OGX distribution instance that can generate configuration
    and documentation files.
    """

    name: str
    description: str
    distro_type: Literal["self_hosted", "remote_hosted", "ondevice"]

    # Now uses BuildProvider for build config, not Provider
    providers: dict[str, list[BuildProvider]]
    run_configs: dict[str, RunConfigSettings]
    template_path: Path | None = None

    # Optional configuration
    run_config_env_vars: dict[str, tuple[str, str]] | None = None
    container_image: str | None = None

    available_models_by_provider: dict[str, list[ProviderModelEntry]] | None = None

    def generate_markdown_docs(self) -> str:
        providers_table = "| API | Provider(s) |\n"
        providers_table += "|-----|-------------|\n"

        for api, providers in sorted(self.providers.items()):
            providers_str = ", ".join(f"`{p.provider_type}`" for p in providers)
            providers_table += f"| {api} | {providers_str} |\n"

        if self.template_path is not None:
            template = self.template_path.read_text()
            comment = "<!-- This file was auto-generated by distro_codegen.py, please edit source -->\n"
            orphantext = "---\norphan: true\n---\n"

            if template.startswith(orphantext):
                template = template.replace(orphantext, orphantext + comment)
            else:
                template = comment + template

            # Render template with rich-generated table
            env = jinja2.Environment(
                trim_blocks=True,
                lstrip_blocks=True,
                # NOTE: autoescape is required to prevent XSS attacks
                autoescape=True,
            )
            template = env.from_string(template)

            default_models = []
            if self.available_models_by_provider:
                has_multiple_providers = len(self.available_models_by_provider.keys()) > 1
                for provider_id, model_entries in self.available_models_by_provider.items():
                    for model_entry in model_entries:
                        doc_parts = []
                        if model_entry.aliases:
                            doc_parts.append(f"aliases: {', '.join(model_entry.aliases)}")
                        if has_multiple_providers:
                            doc_parts.append(f"provider: {provider_id}")

                        default_models.append(
                            DefaultModel(
                                model_id=model_entry.provider_model_id,
                                doc_string=(f"({' -- '.join(doc_parts)})" if doc_parts else ""),
                            )
                        )

            return template.render(
                name=self.name,
                description=self.description,
                providers=self.providers,
                providers_table=providers_table,
                run_config_env_vars=self.run_config_env_vars,
                default_models=default_models,
                run_configs=list(self.run_configs.keys()),
            )
        return ""

    def save_distribution(self, yaml_output_dir: Path, doc_output_dir: Path) -> None:
        def enum_representer(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)

        # Register YAML representer for enums
        yaml.add_representer(ModelType, enum_representer)
        yaml.add_representer(StorageBackendType, enum_representer)
        yaml.add_representer(ConnectorType, enum_representer)
        yaml.SafeDumper.add_representer(ModelType, enum_representer)
        yaml.SafeDumper.add_representer(StorageBackendType, enum_representer)
        yaml.SafeDumper.add_representer(ConnectorType, enum_representer)

        for output_dir in [yaml_output_dir, doc_output_dir]:
            output_dir.mkdir(parents=True, exist_ok=True)

        for yaml_pth, settings in self.run_configs.items():
            run_config = settings.run_config(self.name, self.providers, self.container_image)
            with open(yaml_output_dir / yaml_pth, "w") as f:
                yaml.safe_dump(
                    filter_empty_values(run_config),
                    f,
                    sort_keys=False,
                )

        if self.template_path:
            docs = self.generate_markdown_docs()
            with open(doc_output_dir / f"{self.name}.md", "w") as f:
                f.write(docs if docs.endswith("\n") else docs + "\n")
