# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import importlib.resources
import inspect
import os
import re
import tempfile
from pathlib import Path
from typing import Any, get_type_hints

import yaml
from pydantic import BaseModel

from ogx.core.admin import AdminImpl, AdminImplConfig
from ogx.core.connectors.connectors import ConnectorServiceConfig, ConnectorServiceImpl
from ogx.core.conversations.conversations import ConversationServiceConfig, ConversationServiceImpl
from ogx.core.datatypes import (
    Provider,
    QualifiedModel,
    RerankerModel,
    SafetyConfig,
    StackConfig,
    VectorStoresConfig,
)
from ogx.core.distribution import get_provider_registry
from ogx.core.inspect import DistributionInspectConfig, DistributionInspectImpl
from ogx.core.prompts.prompts import PromptServiceConfig, PromptServiceImpl
from ogx.core.providers import ProviderImpl, ProviderImplConfig
from ogx.core.resolver import ProviderRegistry, resolve_impls
from ogx.core.routing_tables.common import CommonRoutingTableImpl
from ogx.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageBackendConfig,
    StorageConfig,
)
from ogx.core.store.registry import create_dist_registry
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import (
    Api,
    Batches,
    Connectors,
    Conversations,
    Files,
    Inference,
    Inspect,
    Models,
    ModelType,
    Prompts,
    Providers,
    RegisterModelRequest,
    RegisterShieldRequest,
    Responses,
    Safety,
    Shields,
    ToolGroupNotFoundError,
    VectorIO,
)

logger = get_logger(name=__name__, category="core")


class OGX(
    Providers,
    Inference,
    Responses,
    Batches,
    Safety,
    VectorIO,
    Models,
    Shields,
    Inspect,
    Files,
    Prompts,
    Conversations,
    Connectors,
):
    """Composite protocol combining all OGX API interfaces."""

    pass


# Resources to register based on configuration.
# If a request class is specified, the configuration object will be converted to this class before invoking the registration method.
RESOURCES = [
    ("models", Api.models, "register_model", "list_models", RegisterModelRequest),
    ("shields", Api.shields, "register_shield", "list_shields", RegisterShieldRequest),
    ("vector_stores", Api.vector_stores, "register_vector_store", "list_vector_stores", None),
]


REGISTRY_REFRESH_INTERVAL_SECONDS = 300
REGISTRY_REFRESH_TASK = None
TEST_RECORDING_CONTEXT = None

# ID fields for registered resources that should trigger skipping
# when they resolve to empty/None (from conditional env vars like :+)
RESOURCE_ID_FIELDS = [
    "vector_store_id",
    "model_id",
    "shield_id",
]


def is_request_model(t: Any) -> bool:
    """Check if a type is a request model (Pydantic BaseModel)."""
    return inspect.isclass(t) and issubclass(t, BaseModel)


async def invoke_with_optional_request(method: Any) -> Any:
    """Invoke a method, automatically creating a request instance if needed.

    For APIs that use request models, this will create an empty request object.
    For backward compatibility, falls back to calling without arguments.
    """
    try:
        hints = get_type_hints(method)
    except Exception:
        # Forward references can't be resolved, fall back to calling without request
        return await method()

    params = list(inspect.signature(method).parameters.values())
    params = [p for p in params if p.name != "self"]

    if not params:
        return await method()

    # Build arguments for the method call
    args: dict[str, Any] = {}
    can_call = True

    for param in params:
        param_type = hints.get(param.name)

        # If it's a request model, try to create an empty instance
        if param_type and is_request_model(param_type):
            try:
                args[param.name] = param_type()
            except Exception:
                # Request model requires arguments, can't create empty instance
                can_call = False
                break
        # If it has a default value, we can skip it (will use default)
        elif param.default != inspect.Parameter.empty:
            continue
        # Required parameter that's not a request model - can't provide it
        else:
            can_call = False
            break

    if can_call and args:
        return await method(**args)

    # Fall back to calling without arguments for backward compatibility
    return await method()


async def register_resources(run_config: StackConfig, impls: dict[Api, Any]) -> None:
    """Register all resources defined in the run configuration with their respective providers.

    Args:
        run_config: The stack run configuration containing registered_resources.
        impls: Dictionary mapping APIs to their provider implementations.
    """
    for rsrc, api, register_method, list_method, request_class in RESOURCES:
        objects = getattr(run_config.registered_resources, rsrc)
        if api not in impls:
            continue

        method = getattr(impls[api], register_method)
        for obj in objects:
            if hasattr(obj, "provider_id"):
                # Do not register models on disabled providers
                if not obj.provider_id or obj.provider_id == "__disabled__":
                    logger.debug("Skipping registration for disabled provider", resource=rsrc.capitalize())
                    continue
                # Handle provider_id="all" - register unprefixed alias using first active provider
                if obj.provider_id == "all":
                    if rsrc != "models":
                        logger.warning(
                            "provider_id=all is only supported for models, skipping",
                            resource=rsrc.capitalize(),
                        )
                        continue
                    # Get all active inference providers from the routing table
                    routing_table = impls[api]
                    if not hasattr(routing_table, "impls_by_provider_id"):
                        logger.warning(
                            "Cannot resolve provider_id=all - routing table has no providers",
                            resource=rsrc.capitalize(),
                        )
                        continue
                    provider_ids = list(routing_table.impls_by_provider_id.keys())
                    if not provider_ids:
                        logger.warning(
                            "Cannot resolve provider_id=all - no active providers found",
                            resource=rsrc.capitalize(),
                        )
                        continue
                    # Use first active provider for the unprefixed alias
                    first_provider = provider_ids[0]
                    logger.info(
                        "Registering unprefixed model alias using first active inference provider",
                        model_id=obj.model_id,
                        provider_id=first_provider,
                        available_providers=provider_ids,
                    )

                    # Mark this as an unprefixed alias in metadata
                    metadata = (obj.metadata or {}).copy()
                    metadata["_unprefixed_alias"] = True
                    obj_copy = obj.model_copy(
                        update={
                            "provider_id": first_provider,
                            "metadata": metadata,
                        }
                    )

                    if request_class is not None:
                        request = request_class(**obj_copy.model_dump())
                        try:
                            await method(request)
                        except Exception as e:
                            logger.warning(
                                "Failed to register unprefixed model alias",
                                model_id=obj_copy.model_id,
                                provider_id=first_provider,
                                error=str(e),
                            )
                    else:
                        await method(**{k: getattr(obj_copy, k) for k in obj_copy.model_dump().keys()})
                    continue

                logger.debug(
                    "Registering resource for provider",
                    resource=rsrc.capitalize(),
                    obj=obj,
                    provider_id=obj.provider_id,
                )

            # TODO: Once all register methods are migrated to accept request objects,
            # remove this conditional and always use the request_class pattern.
            if request_class is not None:
                request = request_class(**obj.model_dump())
                await method(request)
            else:
                # we want to maintain the type information in arguments to method.
                # instead of method(**obj.model_dump()), which may convert a typed attr to a dict,
                # we use model_dump() to find all the attrs and then getattr to get the still typed
                # value.
                await method(**{k: getattr(obj, k) for k in obj.model_dump().keys()})

        method = getattr(impls[api], list_method)
        response = await invoke_with_optional_request(method)

        objects_to_process = response.data if hasattr(response, "data") else response

        for obj in objects_to_process:
            logger.debug(
                ": served by",
                rsrc_capitalize=rsrc.capitalize(),
                identifier=obj.identifier,
                provider_id=obj.provider_id,
            )


async def auto_register_tool_groups(run_config: StackConfig, impls: dict[Api, Any]) -> None:
    """Auto-register built-in tool groups based on configured tool_runtime providers.

    For each tool_runtime provider whose spec declares a toolgroup_id,
    register that tool group automatically. This replaces the old explicit
    tool_groups config in registered_resources.

    When multiple providers map to the same toolgroup (e.g. brave-search and
    tavily-search both serve builtin::websearch), the provider with a configured
    api_key wins. If none have a key, the first candidate is used so the tool
    group still exists (invocation will fail with a clear provider error rather
    than "tool not found").
    """
    if Api.tool_groups not in impls:
        return

    tool_groups_impl = impls[Api.tool_groups]

    registry = get_provider_registry(run_config)
    type_to_toolgroup = {
        ptype: spec.toolgroup_id for ptype, spec in registry.get(Api.tool_runtime, {}).items() if spec.toolgroup_id
    }

    # Single-pass selection: keep the first provider per toolgroup, but
    # let a provider with a configured api_key replace one without.
    chosen: dict[str, Provider] = {}
    chosen_has_key: dict[str, bool] = {}
    for provider in run_config.providers.get("tool_runtime", []):
        if not provider.provider_id or provider.provider_id == "__disabled__":
            continue
        toolgroup_id = type_to_toolgroup.get(provider.provider_type)
        if not toolgroup_id:
            continue

        has_key = "api_key" not in provider.config or bool(provider.config.get("api_key"))
        if toolgroup_id not in chosen or (has_key and not chosen_has_key[toolgroup_id]):
            chosen[toolgroup_id] = provider
            chosen_has_key[toolgroup_id] = has_key

    for toolgroup_id, provider in chosen.items():
        # Unregister first so register_tool_group rebuilds in-memory tool
        # indexes (_index_tools) which are empty after a restart.
        try:
            await tool_groups_impl.unregister_toolgroup(toolgroup_id)
        except ToolGroupNotFoundError:
            pass

        logger.info("Auto-registering tool group", toolgroup_id=toolgroup_id, provider_id=provider.provider_id)
        await tool_groups_impl.register_tool_group(
            toolgroup_id=toolgroup_id,
            provider_id=provider.provider_id,
        )


async def register_connectors(run_config: StackConfig, impls: dict[Api, Any]) -> None:
    """Register connectors from config"""
    if Api.connectors not in impls:
        return

    connectors_impl = impls[Api.connectors]

    # Get connector IDs from config
    config_connector_ids = {c.connector_id for c in run_config.connectors}

    # Register/Update config connectors
    for connector in run_config.connectors:
        logger.debug("Registering connector", connector_id=connector.connector_id)
        await connectors_impl.register_connector(
            connector_id=connector.connector_id,
            connector_type=connector.connector_type,
            url=connector.url,
            server_label=connector.server_label,
        )

    # Remove connectors not in config (orphan cleanup)
    existing_connectors = await connectors_impl.list_connectors()
    for connector in existing_connectors.data:
        if connector.connector_id not in config_connector_ids:
            logger.info("Removing orphaned connector", connector_id=connector.connector_id)
            await connectors_impl.unregister_connector(connector.connector_id)


async def validate_vector_stores_config(vector_stores_config: VectorStoresConfig | None, impls: dict[Api, Any]) -> None:
    """Validate vector stores configuration."""
    if vector_stores_config is None:
        return

    # Validate default embedding model
    if vector_stores_config.default_embedding_model is not None:
        await _validate_embedding_model(vector_stores_config.default_embedding_model, impls)

    # Validate default reranker model
    if vector_stores_config.default_reranker_model is not None:
        await _validate_reranker_model(vector_stores_config.default_reranker_model, impls)

    # Validate rewrite query params
    if vector_stores_config.rewrite_query_params:
        if vector_stores_config.rewrite_query_params.model:
            await _validate_rewrite_query_model(vector_stores_config.rewrite_query_params.model, impls)


async def _validate_embedding_model(embedding_model: QualifiedModel, impls: dict[Api, Any]) -> None:
    """Validate that an embedding model exists and has required metadata."""
    provider_id = embedding_model.provider_id
    model_id = embedding_model.model_id
    model_identifier = f"{provider_id}/{model_id}"

    if Api.models not in impls:
        raise ValueError(f"Models API is not available but vector_stores config requires model '{model_identifier}'")

    models_impl = impls[Api.models]
    response = await models_impl.list_models()
    models_list = {m.identifier: m for m in response.data if m.model_type == "embedding"}

    model = models_list.get(model_identifier)
    if model is None:
        raise ValueError(
            f"Embedding model '{model_identifier}' not found. Available embedding models: {list(models_list.keys())}"
        )

    # if not in metadata, fetch from config default
    embedding_dimension = model.metadata.get("embedding_dimension", embedding_model.embedding_dimensions)
    if embedding_dimension is None:
        raise ValueError(f"Embedding model '{model_identifier}' is missing 'embedding_dimension' in metadata")

    try:
        int(embedding_dimension)
    except ValueError as err:
        raise ValueError(f"Embedding dimension '{embedding_dimension}' cannot be converted to an integer") from err

    logger.debug(
        "Validated embedding model", model_identifier=model_identifier, embedding_dimension=embedding_dimension
    )


async def _validate_reranker_model(reranker_model: RerankerModel, impls: dict[Api, Any]) -> None:
    """Validate that a reranker model exists."""
    provider_id = reranker_model.provider_id
    model_id = reranker_model.model_id
    model_identifier = f"{provider_id}/{model_id}"

    if Api.models not in impls:
        raise ValueError(f"Models API is not available but vector_stores config requires model '{model_identifier}'")

    models_impl = impls[Api.models]
    response = await models_impl.list_models()
    models_list = {m.identifier: m for m in response.data if m.model_type == ModelType.rerank}

    model = models_list.get(model_identifier)
    if model is None:
        raise ValueError(
            f"Reranker model '{model_identifier}' not found. Available reranker models: {list(models_list.keys())}"
        )

    logger.debug("Validated reranker model", model_identifier=model_identifier)


async def _validate_rewrite_query_model(rewrite_query_model: QualifiedModel, impls: dict[Api, Any]) -> None:
    """Validate that a rewrite query model exists and is accessible."""
    provider_id = rewrite_query_model.provider_id
    model_id = rewrite_query_model.model_id
    model_identifier = f"{provider_id}/{model_id}"

    if Api.models not in impls:
        raise ValueError(
            f"Models API is not available but vector_stores config requires rewrite query model '{model_identifier}'"
        )

    models_impl = impls[Api.models]
    response = await models_impl.list_models()
    llm_models_list = {m.identifier: m for m in response.data if m.model_type == "llm"}

    model = llm_models_list.get(model_identifier)
    if model is None:
        raise ValueError(
            f"Rewrite query model '{model_identifier}' not found. Available LLM models: {list(llm_models_list.keys())}"
        )

    logger.debug("Validated rewrite query model", model_identifier=model_identifier)


async def validate_safety_config(safety_config: SafetyConfig | None, impls: dict[Api, Any]) -> None:
    """Validate that the configured default shield exists among registered shields.

    Args:
        safety_config: Optional safety configuration with a default_shield_id.
        impls: Dictionary mapping APIs to their provider implementations.

    Raises:
        ValueError: If the default shield ID is not found among registered shields.
    """
    if safety_config is None or safety_config.default_shield_id is None:
        return

    if Api.shields not in impls:
        raise ValueError("Safety configuration requires the shields API to be enabled")

    if Api.safety not in impls:
        raise ValueError("Safety configuration requires the safety API to be enabled")

    shields_impl = impls[Api.shields]
    response = await shields_impl.list_shields()
    shields_by_id = {shield.identifier: shield for shield in response.data}

    default_shield_id = safety_config.default_shield_id
    # don't validate if there are no shields registered
    if shields_by_id and default_shield_id not in shields_by_id:
        available = sorted(shields_by_id)
        raise ValueError(
            f"Configured default_shield_id '{default_shield_id}' not found among registered shields."
            f" Available shields: {available}"
        )


class EnvVarError(Exception):
    """Raised when a required environment variable is not set or empty."""

    def __init__(self, var_name: str, path: str = ""):
        self.var_name = var_name
        self.path = path
        super().__init__(
            f"Environment variable '{var_name}' not set or empty {f'at {path}' if path else ''}. "
            f"Use ${{env.{var_name}:=default_value}} to provide a default value, "
            f"${{env.{var_name}:+value_if_set}} to make the field conditional, "
            f"or ensure the environment variable is set."
        )


def replace_env_vars(config: Any, path: str = "") -> Any:
    """Recursively replace environment variable references in a configuration object."""
    if isinstance(config, dict):
        # Special handling for auth provider_config with conditional type field
        # This allows auth to be enabled/disabled via environment variables
        # Example: type: ${env.AUTH_PROVIDER:+oauth2_token}
        if "provider_config" in config and path == "server.auth":
            provider_cfg = config.get("provider_config")
            if isinstance(provider_cfg, dict) and "type" in provider_cfg:
                try:
                    # Resolve the type field first to check if auth should be enabled
                    resolved_type = replace_env_vars(provider_cfg["type"], f"{path}.provider_config.type")

                    # If type is empty/None, disable auth by setting provider_config to None
                    # This prevents validation errors on the discriminated union
                    if resolved_type is None or resolved_type == "":
                        # Process rest of config normally but exclude provider_config from expansion
                        # to avoid EnvVarError from bare env vars (e.g., ${env.KEYCLOAK_URL})
                        result = {
                            k: replace_env_vars(v, f"{path}.{k}" if path else k)
                            for k, v in config.items()
                            if k != "provider_config"
                        }
                        result["provider_config"] = None
                        return result
                except EnvVarError as e:
                    # If we can't resolve type, continue with normal processing
                    # and let validation catch the error
                    logger.debug(
                        "Could not resolve auth provider type field: - continuing with normal processing",
                        var_name=e.var_name,
                    )

        result = {}
        for k, v in config.items():
            try:
                result[k] = replace_env_vars(v, f"{path}.{k}" if path else k)
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result

    elif isinstance(config, list):
        # result is assigned as list here but dict/str in other branches.
        # Mypy cannot track that only one branch executes.
        result = []  # type: ignore[assignment]
        for i, v in enumerate(config):
            try:
                # Special handling for providers: first resolve the provider_id to check if provider
                # is disabled so that we can skip config env variable expansion and avoid validation errors
                if isinstance(v, dict) and "provider_id" in v:
                    try:
                        resolved_provider_id = replace_env_vars(v["provider_id"], f"{path}[{i}].provider_id")
                        if resolved_provider_id == "__disabled__":
                            logger.debug(
                                "Skipping config env variable expansion for disabled provider",
                                v_get_provider_id=v.get("provider_id", ""),
                            )
                            continue
                    except EnvVarError:
                        # If we can't resolve the provider_id, continue with normal processing
                        pass

                # Special handling for registered resources: check if ID field resolves to empty/None
                # from conditional env vars (e.g., ${env.VAR:+value}) and skip the entry if so
                if isinstance(v, dict):
                    should_skip = False
                    for id_field in RESOURCE_ID_FIELDS:
                        if id_field in v:
                            try:
                                resolved_id = replace_env_vars(v[id_field], f"{path}[{i}].{id_field}")
                                if resolved_id is None or resolved_id == "":
                                    logger.debug(
                                        "Skipping [] with empty (conditional env var not set)",
                                        path=path,
                                        i=i,
                                        id_field=id_field,
                                    )
                                    should_skip = True
                                    break
                            except EnvVarError as e:
                                logger.warning(
                                    "Could not resolve in [], env var",
                                    id_field=id_field,
                                    path=path,
                                    i=i,
                                    var_name=e.var_name,
                                    error=str(e),
                                )
                    if should_skip:
                        continue

                # Normal processing
                # result is a list here, but mypy sees it could be dict/str
                result.append(replace_env_vars(v, f"{path}[{i}]"))  # type: ignore[attr-defined]
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result

    elif isinstance(config, str):
        # Pattern supports bash-like syntax: := for default and :+ for conditional and a optional value
        pattern = r"\${env\.([A-Z0-9_]+)(?::([=+])([^}]*))?}"

        def get_env_var(match: re.Match):
            env_var = match.group(1)
            operator = match.group(2)  # '=' for default, '+' for conditional
            value_expr = match.group(3)

            env_value = os.environ.get(env_var)

            if operator == "=":  # Default value syntax: ${env.FOO:=default}
                # If the env is set like ${env.FOO:=default} then use the env value when set
                if env_value:
                    value = env_value
                else:
                    # If the env is not set, look for a default value
                    # value_expr returns empty string (not None) when not matched
                    # This means ${env.FOO:=} and it's accepted and returns empty string - just like bash
                    if value_expr == "":
                        return ""
                    else:
                        value = value_expr

            elif operator == "+":  # Conditional value syntax: ${env.FOO:+value_if_set}
                # If the env is set like ${env.FOO:+value_if_set} then use the value_if_set
                if env_value:
                    if value_expr:
                        value = value_expr
                    # This means ${env.FOO:+}
                    else:
                        # Just like bash, this doesn't care whether the env is set or not and applies
                        # the value, in this case the empty string
                        return ""
                else:
                    # Just like bash, this doesn't care whether the env is set or not, since it's not set
                    # we return an empty string
                    value = ""
            else:  # No operator case: ${env.FOO}
                if not env_value:
                    raise EnvVarError(env_var, path)
                value = env_value

            # expand "~" from the values
            return os.path.expanduser(value)

        try:
            # re.sub returns str, but result could be dict/list in other branches
            result = re.sub(pattern, get_env_var, config)  # type: ignore[assignment]
            # Only apply type conversion if substitution actually happened
            if result != config:
                # result is str here but mypy sees it could be dict/list
                return _convert_string_to_proper_type(result)  # type: ignore[arg-type]
            return result
        except EnvVarError as e:
            raise EnvVarError(e.var_name, e.path) from None

    return config


def _convert_string_to_proper_type(value: str) -> Any:
    # This might be tricky depending on what the config type is, if  'str | None' we are
    # good, if 'str' we need to keep the empty string... 'str | None' is more common and
    # providers config should be typed this way.
    # TODO: we could try to load the config class and see if the config has a field with type 'str | None'
    # and then convert the empty string to None or not
    if value == "":
        return None

    lowered = value.lower()
    if lowered == "true":
        return True
    elif lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def cast_distro_name_to_string(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure that any value for a key 'distro_name' in a config_dict is a string"""
    if "distro_name" in config_dict and config_dict["distro_name"] is not None:
        config_dict["distro_name"] = str(config_dict["distro_name"])
    return config_dict


def add_internal_implementations(impls: dict[Api, Any], config: StackConfig, policy: list) -> None:
    """Add internal implementations (inspect, providers, admin, etc.) to the implementations dictionary."""
    # deps expects dict[str, Any] but receives dict[Api, Any].
    # Api is an enum, runtime compatible as dict key.
    inspect_impl = DistributionInspectImpl(
        DistributionInspectConfig(config=config),
        deps=impls,  # type: ignore[arg-type]
    )
    impls[Api.inspect] = inspect_impl

    providers_impl = ProviderImpl(
        ProviderImplConfig(config=config),
        deps=impls,  # type: ignore[arg-type]
    )
    impls[Api.providers] = providers_impl

    admin_impl = AdminImpl(
        AdminImplConfig(config=config),
        deps=impls,  # type: ignore[arg-type]
    )
    impls[Api.admin] = admin_impl

    prompts_impl = PromptServiceImpl(
        PromptServiceConfig(config=config),
        deps=impls,
    )
    impls[Api.prompts] = prompts_impl

    conversations_impl = ConversationServiceImpl(
        ConversationServiceConfig(config=config, policy=policy),
        deps=impls,
    )
    impls[Api.conversations] = conversations_impl

    connectors_impl = ConnectorServiceImpl(
        ConnectorServiceConfig(config=config),
    )
    impls[Api.connectors] = connectors_impl


def _initialize_storage(run_config: StackConfig):
    kv_backends: dict[str, StorageBackendConfig] = {}
    sql_backends: dict[str, StorageBackendConfig] = {}
    for backend_name, backend_config in run_config.storage.backends.items():
        type = backend_config.type.value
        if type.startswith("kv_"):
            kv_backends[backend_name] = backend_config
        elif type.startswith("sql_"):
            sql_backends[backend_name] = backend_config
        else:
            raise ValueError(f"Unknown storage backend type: {type}")

    from ogx.core.storage.kvstore.kvstore import register_kvstore_backends
    from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends

    register_kvstore_backends(kv_backends)
    register_sqlstore_backends(sql_backends)


class Stack:
    """Manages the lifecycle of a OGX instance, including initialization, registry refresh, and shutdown."""

    def __init__(self, run_config: StackConfig, provider_registry: ProviderRegistry | None = None):
        self.run_config = run_config
        self.provider_registry = provider_registry
        self.impls = None

    # Produces a stack of providers for the given run config. Not all APIs may be
    # asked for in the run config.
    async def initialize(self):
        if "OGX_TEST_INFERENCE_MODE" in os.environ:
            from ogx.testing.api_recorder import setup_api_recording

            global TEST_RECORDING_CONTEXT
            TEST_RECORDING_CONTEXT = setup_api_recording()
            if TEST_RECORDING_CONTEXT:
                TEST_RECORDING_CONTEXT.__enter__()
                logger.info("API recording enabled", mode=os.environ.get("OGX_TEST_INFERENCE_MODE"))

        _initialize_storage(self.run_config)
        stores = self.run_config.storage.stores
        if not stores.metadata:
            raise ValueError("storage.stores.metadata must be configured with a kv_* backend")
        dist_registry, _ = await create_dist_registry(stores.metadata, self.run_config.distro_name)
        policy = self.run_config.server.auth.access_policy if self.run_config.server.auth else []

        internal_impls = {}
        add_internal_implementations(internal_impls, self.run_config, policy)

        impls = await resolve_impls(
            self.run_config,
            self.provider_registry or get_provider_registry(self.run_config),
            dist_registry,
            policy,
            internal_impls,
        )

        if Api.prompts in impls:
            await impls[Api.prompts].initialize()
        if Api.conversations in impls:
            await impls[Api.conversations].initialize()
        if Api.connectors in impls:
            await impls[Api.connectors].initialize()

        await register_resources(self.run_config, impls)
        await auto_register_tool_groups(self.run_config, impls)
        await register_connectors(self.run_config, impls)
        await refresh_registry_once(impls)
        await validate_vector_stores_config(self.run_config.vector_stores, impls)
        await validate_safety_config(self.run_config.safety, impls)
        self.impls = impls

    def create_registry_refresh_task(self):
        assert self.impls is not None, "Must call initialize() before starting"

        global REGISTRY_REFRESH_TASK
        interval = self.run_config.server.registry_refresh_interval_seconds
        REGISTRY_REFRESH_TASK = asyncio.create_task(refresh_registry_task(self.impls, interval))

        def cb(task):
            import traceback

            if task.cancelled():
                logger.warning("Model refresh task cancelled")
            elif task.exception():
                logger.error("Model refresh task failed", error=str(task.exception()))
                traceback.print_exception(task.exception())
            else:
                logger.debug("Model refresh task completed")

        REGISTRY_REFRESH_TASK.add_done_callback(cb)

    async def shutdown(self):
        for impl in self.impls.values():
            impl_name = impl.__class__.__name__
            logger.debug("Shutting down", impl_name=impl_name)
            try:
                if hasattr(impl, "shutdown"):
                    await asyncio.wait_for(impl.shutdown(), timeout=5)
                else:
                    logger.warning("No shutdown method for", impl_name=impl_name)
            except TimeoutError:
                logger.exception("Shutdown timeout", impl_name=impl_name)
            except (Exception, asyncio.CancelledError) as e:
                logger.exception("Failed to shutdown", impl_name=impl_name, error=str(e))

        global TEST_RECORDING_CONTEXT
        if TEST_RECORDING_CONTEXT:
            try:
                TEST_RECORDING_CONTEXT.__exit__(None, None, None)
            except Exception as e:
                logger.error("Error during API recording cleanup", error=str(e))

        global REGISTRY_REFRESH_TASK
        if REGISTRY_REFRESH_TASK:
            REGISTRY_REFRESH_TASK.cancel()

        # Shutdown storage backends
        from ogx.core.storage.kvstore.kvstore import shutdown_kvstore_backends
        from ogx.core.storage.sqlstore.sqlstore import shutdown_sqlstore_backends

        try:
            await shutdown_kvstore_backends()
        except Exception as e:
            logger.exception("Failed to shutdown KV store backends", error=str(e))

        try:
            await shutdown_sqlstore_backends()
        except Exception as e:
            logger.exception("Failed to shutdown SQL store backends", error=str(e))


async def refresh_registry_once(impls: dict[Api, Any]):
    """Refresh all routing table registries once by calling their refresh methods."""
    logger.debug("refreshing registry")
    routing_tables = [v for v in impls.values() if isinstance(v, CommonRoutingTableImpl)]
    for routing_table in routing_tables:
        await routing_table.refresh()


async def refresh_registry_task(impls: dict[Api, Any], interval_seconds: int = REGISTRY_REFRESH_INTERVAL_SECONDS):
    """Background task that periodically refreshes routing table registries."""
    logger.info("starting registry refresh task", interval_seconds=interval_seconds)
    while True:
        await refresh_registry_once(impls)

        await asyncio.sleep(interval_seconds)


def get_stack_run_config_from_distro(distro: str) -> StackConfig:
    """Load a StackConfig from a named distribution's bundled config.yaml."""
    distro_path = importlib.resources.files("ogx") / f"distributions/{distro}/config.yaml"

    with importlib.resources.as_file(distro_path) as path:
        if not path.exists():
            raise ValueError(f"Distribution '{distro}' not found at {distro_path}")
        run_config = yaml.safe_load(path.open())

    return StackConfig(**replace_env_vars(run_config))


def run_config_from_dynamic_config_spec(
    dynamic_config_spec: str,
    provider_registry: ProviderRegistry | None = None,
    distro_dir: Path | None = None,
    distro_name: str = "dynamic-distro",
) -> StackConfig:
    """
    Create a dynamic distribution from a list of API providers.

    The list should be of the form "api=provider", e.g. "inference=fireworks". If you have
    multiple pairs, separate them with commas or semicolons, e.g. "inference=fireworks,safety=llama-guard,agents=builtin"

    You can optionally specify config parameters using URL query parameter syntax,
    e.g. "inference=inline::sentence-transformers?trust_remote_code=true&max_seq_length=512"
    """

    api_providers = dynamic_config_spec.replace(";", ",").split(",")
    provider_registry = get_provider_registry() if provider_registry is None else provider_registry

    distro_dir = distro_dir or Path(tempfile.mkdtemp())
    # Explicit type annotation for better type inference in the loop below
    provider_configs_by_api: dict[str, Any] = {}
    for api_provider in api_providers:
        if "=" not in api_provider:
            raise ValueError(
                f"Failed to parse provider spec '{api_provider}'. Expected format: api=provider (e.g. inference=fireworks)"
            )
        api_str, provider_with_params = api_provider.split("=", 1)

        # Parse provider name and optional config parameters
        # Format: provider_name?param1=value1&param2=value2
        if "?" in provider_with_params:
            provider, params_str = provider_with_params.split("?", 1)
            config_overrides = {}
            # Parse key=value pairs separated by &
            # Values are kept as strings and Pydantic will handle type conversion
            for param in params_str.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    config_overrides[key] = value
        else:
            provider = provider_with_params
            config_overrides = {}

        try:
            api = Api(api_str)
        except ValueError:
            raise ValueError(f"Failed to parse provider spec: '{api_str}' is not a valid API") from None

        providers_by_type = provider_registry.get(api)
        if providers_by_type is None:
            raise ValueError(f"Failed to find providers for API '{api_str}'")
        provider_spec = providers_by_type.get(provider)
        if not provider_spec:
            provider_spec = providers_by_type.get(f"inline::{provider}")
        if not provider_spec:
            provider_spec = providers_by_type.get(f"remote::{provider}")

        if not provider_spec:
            raise ValueError(
                f"Provider {provider} (or remote::{provider} or inline::{provider}) not found for API {api}"
            )

        # call method "sample_run_config" on the provider spec config class
        provider_config_type = instantiate_class_type(provider_spec.config_class)
        provider_config = replace_env_vars(provider_config_type.sample_run_config(__distro_dir__=str(distro_dir)))

        # Apply config overrides
        provider_config.update(config_overrides)

        provider_configs_by_api.setdefault(api_str, []).append(
            Provider(
                provider_id=provider_spec.provider_type.split("::")[-1],
                provider_type=provider_spec.provider_type,
                config=provider_config,
            )
        )
    config = StackConfig(
        distro_name=distro_name,
        apis=list(provider_configs_by_api.keys()),
        providers=provider_configs_by_api,
        storage=StorageConfig(
            backends={
                "kv_default": SqliteKVStoreConfig(
                    db_path=f"${{env.SQLITE_STORE_DIR:={distro_dir}}}/kvstore.db",
                ),
                "sql_default": SqliteSqlStoreConfig(
                    db_path=f"${{env.SQLITE_STORE_DIR:={distro_dir}}}/sql_store.db",
                ),
            },
            stores=ServerStoresConfig(
                metadata=KVStoreReference(backend="kv_default", namespace="registry"),
                inference=InferenceStoreReference(backend="sql_default", table_name="inference_store"),
                conversations=SqlStoreReference(backend="sql_default", table_name="openai_conversations"),
                prompts=KVStoreReference(backend="kv_default", namespace="prompts"),
                connectors=KVStoreReference(backend="kv_default", namespace="connectors"),
            ),
        ),
    )
    return config
