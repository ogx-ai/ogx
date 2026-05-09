# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import glob
import importlib
import os
from typing import Any

import yaml
from pydantic import BaseModel

from ogx.core.datatypes import StackConfig
from ogx.core.external import load_external_apis
from ogx.log import get_logger
from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)

logger = get_logger(name=__name__, category="core")


INTERNAL_APIS = {Api.inspect, Api.providers, Api.prompts, Api.conversations, Api.connectors, Api.admin}


def stack_apis() -> list[Api]:
    """Return all available API types.

    Returns:
        A list of all Api enum members.
    """
    return list(Api)


class AutoRoutedApiInfo(BaseModel):
    """Pairing of a routing table API with its corresponding router API for automatic routing."""

    routing_table_api: Api
    router_api: Api


def builtin_automatically_routed_apis() -> list[AutoRoutedApiInfo]:
    """Return the built-in routing table to router API pairings.

    Returns:
        A list of AutoRoutedApiInfo objects mapping routing table APIs to their router APIs.
    """
    return [
        AutoRoutedApiInfo(
            routing_table_api=Api.models,
            router_api=Api.inference,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.shields,
            router_api=Api.safety,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.tool_groups,
            router_api=Api.tool_runtime,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.vector_stores,
            router_api=Api.vector_io,
        ),
    ]


def providable_apis() -> list[Api]:
    """Return the list of APIs that can have external providers configured.

    Returns:
        APIs excluding internal APIs and routing table APIs that are auto-generated.
    """
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    return [api for api in Api if api not in routing_table_apis and api not in INTERNAL_APIS]


def _load_remote_provider_spec(spec_data: dict[str, Any], api: Api) -> ProviderSpec:
    spec = RemoteProviderSpec(api=api, provider_type=f"remote::{spec_data['adapter_type']}", **spec_data)
    return spec


def _load_inline_provider_spec(spec_data: dict[str, Any], api: Api, provider_name: str) -> ProviderSpec:
    spec = InlineProviderSpec(api=api, provider_type=f"inline::{provider_name}", **spec_data)
    return spec


def get_provider_registry(
    config: StackConfig | None = None, listing: bool = False
) -> dict[Api, dict[str, ProviderSpec]]:
    """Get the provider registry, optionally including external providers.

    This function loads both built-in providers and external providers from YAML files or from their provided modules.
    External providers are loaded from a directory structure like:

    providers.d/
      remote/
        inference/
          custom_ollama.yaml
          vllm.yaml
        vector_io/
          qdrant.yaml
        safety/
          llama-guard.yaml
      inline/
        inference/
          custom_ollama.yaml
          vllm.yaml
        vector_io/
          qdrant.yaml
        safety/
          llama-guard.yaml

    This method is overloaded in that it can be called from a variety of places: during list-deps, during run, during stack construction.
    So when listing external providers from a module, there are scenarios where the pip package required to import the module might not be available yet.
    There is special handling for all of the potential cases this method can be called from.

    Args:
        config: Optional object containing the external providers directory path
        listing: Optional bool delineating whether or not this is being called from a list-deps process

    Returns:
        A dictionary mapping APIs to their available providers

    Raises:
        FileNotFoundError: If the external providers directory doesn't exist
        ValueError: If any provider spec is invalid
    """

    registry: dict[Api, dict[str, ProviderSpec]] = {}
    for api in providable_apis():
        name = api.name.lower()
        logger.debug("Importing module", name=name)
        try:
            module = importlib.import_module(f"ogx.providers.registry.{name}")
            registry[api] = {a.provider_type: a for a in module.available_providers()}
        except ImportError as e:
            logger.warning("Failed to import module", name=name, error=str(e))

    # Refresh providable APIs with external APIs if any
    external_apis = load_external_apis(config)
    for api, api_spec in external_apis.items():
        name = api_spec.name.lower()
        logger.info("Importing external API module", name=name, module=api_spec.module)
        try:
            module = importlib.import_module(api_spec.module)
            registry[api] = {a.provider_type: a for a in module.available_providers()}
        except (ImportError, AttributeError) as e:
            # Populate the registry with an empty dict to avoid breaking the provider registry
            # This assume that the in-tree provider(s) are not available for this API which means
            # that users will need to use external providers for this API.
            registry[api] = {}
            logger.error(
                "Failed to import external API, could not populate in-tree provider registry. "
                "Install the API package to load any in-tree providers for this API.",
                api_name=name,
                api=api.name,
                error=str(e),
            )

    # Check if config has external providers
    if config:
        if hasattr(config, "external_providers_dir") and config.external_providers_dir:
            registry = get_external_providers_from_dir(registry, config)
        # else lets check for modules in each provider
        registry = get_external_providers_from_module(
            registry=registry,
            config=config,
            listing=listing,
        )

    return registry


def get_external_providers_from_dir(
    registry: dict[Api, dict[str, ProviderSpec]], config: StackConfig
) -> dict[Api, dict[str, ProviderSpec]]:
    """Load external provider specs from YAML files in the external providers directory.

    Args:
        registry: Existing provider registry to extend.
        config: Stack configuration containing the external_providers_dir path.

    Returns:
        The updated provider registry with external providers added.
    """
    logger.warning(
        "Specifying external providers via `external_providers_dir` is being deprecated. Please specify `module:` in the provider instead."
    )
    assert config.external_providers_dir is not None, "external_providers_dir must not be None"
    external_providers_dir = os.path.abspath(os.path.expanduser(str(config.external_providers_dir)))
    if not os.path.exists(external_providers_dir):
        raise FileNotFoundError(f"External providers directory not found: {external_providers_dir}")
    logger.info("Loading external providers from", external_providers_dir=external_providers_dir)

    for api in providable_apis():
        api_name = api.name.lower()

        # Process both remote and inline providers
        for provider_type in ["remote", "inline"]:
            api_dir = os.path.join(external_providers_dir, provider_type, api_name)
            if not os.path.exists(api_dir):
                logger.debug("No provider directory found for", provider_type=provider_type, api_name=api_name)
                continue

            # Look for provider spec files in the API directory
            for spec_path in glob.glob(os.path.join(api_dir, "*.yaml")):
                provider_name = os.path.splitext(os.path.basename(spec_path))[0]
                logger.info("Loading provider spec from", provider_type=provider_type, spec_path=spec_path)

                try:
                    with open(spec_path) as f:
                        spec_data = yaml.safe_load(f)

                    if provider_type == "remote":
                        spec = _load_remote_provider_spec(spec_data, api)
                        provider_type_key = f"remote::{provider_name}"
                    else:
                        spec = _load_inline_provider_spec(spec_data, api, provider_name)
                        provider_type_key = f"inline::{provider_name}"

                    logger.info(
                        "Loaded provider spec",
                        provider_type=provider_type,
                        provider_type_key=provider_type_key,
                        spec_path=spec_path,
                    )
                    if provider_type_key in registry[api]:
                        logger.warning(
                            "Overriding already registered provider for",
                            provider_type_key=provider_type_key,
                            name=api.name,
                        )
                    registry[api][provider_type_key] = spec
                    logger.info("Successfully loaded external provider", provider_type_key=provider_type_key)
                except yaml.YAMLError as yaml_err:
                    logger.error("Failed to parse YAML file", spec_path=spec_path, error=str(yaml_err))
                    raise yaml_err
                except Exception as e:
                    logger.error("Failed to load provider spec from", spec_path=spec_path, error=str(e))
                    raise e

    return registry


def get_external_providers_from_module(
    registry: dict[Api, dict[str, ProviderSpec]], config: StackConfig, listing: bool
) -> dict[Api, dict[str, ProviderSpec]]:
    """Load external provider specs from Python modules specified in provider configurations.

    Args:
        registry: Existing provider registry to extend.
        config: Stack configuration containing providers with module references.
        listing: Whether this is being called from a dependency listing process.

    Returns:
        The updated provider registry with module-based external providers added.
    """
    provider_list = None
    provider_list = config.providers.items()
    if provider_list is None:
        logger.warning("Could not get list of providers from config")
        return registry
    for provider_api, providers in provider_list:
        for provider in providers:
            if not hasattr(provider, "module") or provider.module is None:
                continue
            # get provider using module
            try:
                if not listing:
                    package_name = provider.module.split("==")[0]
                    module = importlib.import_module(f"{package_name}.provider")
                    # if config class is wrong you will get an error saying module could not be imported
                    spec = module.get_provider_spec()
                else:
                    # pass in a partially filled out provider spec to satisfy the registry -- knowing we will be overwriting it later upon list-deps and run
                    # in the case we are listing we CANNOT import this module of course because it has not been installed.
                    spec = ProviderSpec(
                        api=Api(provider_api),
                        provider_type=provider.provider_type,
                        is_external=True,
                        module=provider.module,
                        config_class="",
                    )
                provider_type = provider.provider_type
                if isinstance(spec, list):
                    # optionally allow people to pass inline and remote provider specs as a returned list.
                    # with the old method, users could pass in directories of specs using overlapping code
                    # we want to ensure we preserve that flexibility in this method.
                    logger.info(
                        "Detected a list of external provider specs from adding all to the registry",
                        module=provider.module,
                    )
                    for provider_spec in spec:
                        if provider_spec.provider_type != provider.provider_type:
                            continue
                        logger.info("Adding to registry", provider_type=provider.provider_type)
                        registry[Api(provider_api)][provider.provider_type] = provider_spec
                else:
                    registry[Api(provider_api)][provider_type] = spec
            except ModuleNotFoundError as exc:
                raise ValueError(
                    "get_provider_spec not found. If specifying an external provider via `module` in the Provider spec, the Provider must have the `provider.get_provider_spec` module available"
                ) from exc
            except Exception as e:
                logger.error("Failed to load provider spec from module", module=provider.module, error=str(e))
                raise e
    return registry
