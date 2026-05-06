# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import importlib.metadata
import inspect
from typing import Any

from ogx.core.client import get_client_impl
from ogx.core.datatypes import (
    AccessRule,
    AutoRoutedProviderSpec,
    Provider,
    RoutingTableProviderSpec,
    StackConfig,
)
from ogx.core.distribution import builtin_automatically_routed_apis
from ogx.core.external import load_external_apis
from ogx.core.store import DistributionRegistry
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.log import get_logger
from ogx_api import (
    OGX_API_V1ALPHA,
    Admin,
    Api,
    Batches,
    Connectors,
    Conversations,
    ExternalApiSpec,
    FileProcessors,
    Files,
    Inference,
    InferenceProvider,
    Inspect,
    Interactions,
    Messages,
    Models,
    ModelsProtocolPrivate,
    Prompts,
    ProviderSpec,
    RemoteProviderConfig,
    RemoteProviderSpec,
    Responses,
    Safety,
    Shields,
    ShieldsProtocolPrivate,
    ToolGroups,
    ToolGroupsProtocolPrivate,
    ToolRuntime,
    VectorIO,
    VectorStore,
)
from ogx_api import (
    Providers as ProvidersAPI,
)

logger = get_logger(name=__name__, category="core")


class InvalidProviderError(Exception):
    """Raised when a provider is invalid or has been deprecated with an error."""

    pass


def api_protocol_map(external_apis: dict[Api, ExternalApiSpec] | None = None) -> dict[Api, Any]:
    """Get a mapping of API types to their protocol classes.

    Args:
        external_apis: Optional dictionary of external API specifications

    Returns:
        Dictionary mapping API types to their protocol classes
    """
    protocols = {
        Api.admin: Admin,
        Api.providers: ProvidersAPI,
        Api.responses: Responses,
        Api.inference: Inference,
        Api.inspect: Inspect,
        Api.batches: Batches,
        Api.vector_io: VectorIO,
        Api.vector_stores: VectorStore,
        Api.models: Models,
        Api.safety: Safety,
        Api.shields: Shields,
        Api.tool_groups: ToolGroups,
        Api.tool_runtime: ToolRuntime,
        Api.files: Files,
        Api.prompts: Prompts,
        Api.conversations: Conversations,
        Api.file_processors: FileProcessors,
        Api.connectors: Connectors,
        Api.messages: Messages,
        Api.interactions: Interactions,
    }

    if external_apis:
        for api, api_spec in external_apis.items():
            try:
                module = importlib.import_module(api_spec.module)
                api_class = getattr(module, api_spec.protocol)

                protocols[api] = api_class
            except (ImportError, AttributeError):
                logger.exception("Failed to load external API", api_name=api_spec.name)

    return protocols


def api_protocol_map_for_compliance_check(config: Any) -> dict[Api, Any]:
    """Get the API-to-protocol mapping used for provider compliance checks.

    Args:
        config: Stack configuration for loading external APIs.

    Returns:
        Dictionary mapping APIs to their protocol classes, with InferenceProvider replacing Inference.
    """
    external_apis = load_external_apis(config)
    return {
        **api_protocol_map(external_apis),
        Api.inference: InferenceProvider,
    }


def additional_protocols_map() -> dict[Api, Any]:
    """Get the mapping of APIs to their additional private protocol classes for routing table support.

    Returns:
        Dictionary mapping router APIs to tuples of (private_protocol, public_protocol, routing_table_api).
    """
    return {
        Api.inference: (ModelsProtocolPrivate, Models, Api.models),
        Api.tool_groups: (ToolGroupsProtocolPrivate, ToolGroups, Api.tool_groups),
        Api.safety: (ShieldsProtocolPrivate, Shields, Api.shields),
    }


# TODO: make all this naming far less atrocious. Provider. ProviderSpec. ProviderWithSpec. WTF!
class ProviderWithSpec(Provider):
    """A Provider paired with its resolved ProviderSpec for instantiation."""

    spec: ProviderSpec


ProviderRegistry = dict[Api, dict[str, ProviderSpec]]


async def resolve_impls(
    run_config: StackConfig,
    provider_registry: ProviderRegistry,
    dist_registry: DistributionRegistry,
    policy: list[AccessRule],
    internal_impls: dict[Api, Any] | None = None,
) -> dict[Api, Any]:
    """
    Resolves provider implementations by:
    1. Validating and organizing providers.
    2. Sorting them in dependency order.
    3. Instantiating them with required dependencies.
    """
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    router_apis = {x.router_api for x in builtin_automatically_routed_apis()}

    providers_with_specs = validate_and_prepare_providers(
        run_config, provider_registry, routing_table_apis, router_apis
    )

    apis_to_serve = run_config.apis or set(
        list(providers_with_specs.keys()) + [x.value for x in routing_table_apis] + [x.value for x in router_apis]
    )

    providers_with_specs.update(specs_for_autorouted_apis(apis_to_serve))

    sorted_providers = sort_providers_by_deps(providers_with_specs, run_config)

    return await instantiate_providers(sorted_providers, router_apis, dist_registry, run_config, policy, internal_impls)


def specs_for_autorouted_apis(apis_to_serve: list[str] | set[str]) -> dict[str, dict[str, ProviderWithSpec]]:
    """Generates specifications for automatically routed APIs."""
    specs = {}
    for info in builtin_automatically_routed_apis():
        if info.router_api.value not in apis_to_serve:
            continue

        specs[info.routing_table_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__routing_table__",
                provider_type="__routing_table__",
                config={},
                spec=RoutingTableProviderSpec(
                    api=info.routing_table_api,
                    router_api=info.router_api,
                    module="ogx.core.routers",
                    api_dependencies=[],
                    deps__=[f"inner-{info.router_api.value}"],
                ),
            )
        }

        # Add inference as an optional dependency for vector_io to enable query rewriting
        optional_deps = []
        deps_list = [info.routing_table_api.value]
        if info.router_api == Api.vector_io:
            optional_deps = [Api.inference]
            deps_list.append(Api.inference.value)

        specs[info.router_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__autorouted__",
                provider_type="__autorouted__",
                config={},
                spec=AutoRoutedProviderSpec(
                    api=info.router_api,
                    module="ogx.core.routers",
                    routing_table_api=info.routing_table_api,
                    api_dependencies=[info.routing_table_api],
                    optional_api_dependencies=optional_deps,
                    deps__=deps_list,
                ),
            )
        }
    return specs


def validate_and_prepare_providers(
    run_config: StackConfig, provider_registry: ProviderRegistry, routing_table_apis: set[Api], router_apis: set[Api]
) -> dict[str, dict[str, ProviderWithSpec]]:
    """Validates providers, handles deprecations, and organizes them into a spec dictionary."""
    providers_with_specs: dict[str, dict[str, ProviderWithSpec]] = {}

    for api_str, providers in run_config.providers.items():
        api = Api(api_str)
        if api in routing_table_apis:
            raise ValueError(f"Provider for `{api_str}` is automatically provided and cannot be overridden")

        specs = {}
        for provider in providers:
            if not provider.provider_id or provider.provider_id == "__disabled__":
                logger.debug("Provider is disabled", provider_type=provider.provider_type, api=str(api))
                continue

            validate_provider(provider, api, provider_registry)
            p = provider_registry[api][provider.provider_type]
            p.deps__ = [a.value for a in p.api_dependencies] + [a.value for a in p.optional_api_dependencies]
            spec = ProviderWithSpec(spec=p, **provider.model_dump())
            specs[provider.provider_id] = spec

        key = api_str if api not in router_apis else f"inner-{api_str}"
        providers_with_specs[key] = specs

    return providers_with_specs


def validate_provider(provider: Provider, api: Api, provider_registry: ProviderRegistry):
    """Validates if the provider is allowed and handles deprecations."""
    if provider.provider_type not in provider_registry[api]:
        raise ValueError(f"Provider `{provider.provider_type}` is not available for API `{api}`")

    p = provider_registry[api][provider.provider_type]
    if p.deprecation_error:
        logger.error(p.deprecation_error)
        raise InvalidProviderError(p.deprecation_error)
    elif p.deprecation_warning:
        logger.warning(
            "Provider is deprecated and will be removed in a future release",
            provider_type=provider.provider_type,
            api=str(api),
            deprecation=p.deprecation_warning,
        )


def sort_providers_by_deps(
    providers_with_specs: dict[str, dict[str, ProviderWithSpec]], run_config: StackConfig
) -> list[tuple[str, ProviderWithSpec]]:
    """Sorts providers based on their dependencies."""
    sorted_providers: list[tuple[str, ProviderWithSpec]] = topological_sort(
        {k: list(v.values()) for k, v in providers_with_specs.items()}
    )

    logger.debug("Resolved providers", count=len(sorted_providers))
    for api_str, provider in sorted_providers:
        logger.debug("Provider mapping", api=api_str, provider_id=provider.provider_id)
    return sorted_providers


async def instantiate_providers(
    sorted_providers: list[tuple[str, ProviderWithSpec]],
    router_apis: set[Api],
    dist_registry: DistributionRegistry,
    run_config: StackConfig,
    policy: list[AccessRule],
    internal_impls: dict[Api, Any] | None = None,
) -> dict[Api, Any]:
    """Instantiates providers asynchronously while managing dependencies."""
    impls: dict[Api, Any] = internal_impls.copy() if internal_impls else {}
    inner_impls_by_provider_id: dict[str, dict[str, Any]] = {f"inner-{x.value}": {} for x in router_apis}
    for api_str, provider in sorted_providers:
        # Skip providers that are not enabled
        if provider.provider_id is None:
            continue

        try:
            deps = {a: impls[a] for a in provider.spec.api_dependencies}
        except KeyError as e:
            missing_api = e.args[0]
            raise RuntimeError(
                f"Failed to resolve '{provider.spec.api.value}' provider '{provider.provider_id}' of type '{provider.spec.provider_type}': "
                f"required dependency '{missing_api.value}' is not available. "
                f"Please add a '{missing_api.value}' provider to your configuration or check if the provider is properly configured."
            ) from e
        for a in provider.spec.optional_api_dependencies:
            if a in impls:
                deps[a] = impls[a]

        inner_impls = {}
        if isinstance(provider.spec, RoutingTableProviderSpec):
            inner_impls = inner_impls_by_provider_id[f"inner-{provider.spec.router_api.value}"]

        impl = await instantiate_provider(provider, deps, inner_impls, dist_registry, run_config, policy)

        if api_str.startswith("inner-"):
            inner_impls_by_provider_id[api_str][provider.provider_id] = impl
        else:
            api = Api(api_str)
            impls[api] = impl

    # Post-instantiation: Inject VectorIORouter into VectorStoresRoutingTable
    if Api.vector_io in impls and Api.vector_stores in impls:
        vector_io_router = impls[Api.vector_io]
        vector_stores_routing_table = impls[Api.vector_stores]
        if hasattr(vector_stores_routing_table, "vector_io_router"):
            vector_stores_routing_table.vector_io_router = vector_io_router

    return impls


def topological_sort(
    providers_with_specs: dict[str, list[ProviderWithSpec]],
) -> list[tuple[str, ProviderWithSpec]]:
    """Sort providers in dependency order using topological sort.

    Args:
        providers_with_specs: Dictionary mapping API names to their providers with specs.

    Returns:
        A flattened list of (api_name, provider) tuples in dependency order.
    """

    def dfs(kv, visited: set[str], stack: list[str]):
        api_str, providers = kv
        visited.add(api_str)

        deps = []
        for provider in providers:
            for dep in provider.spec.deps__:
                deps.append(dep)

        for dep in deps:
            if dep not in visited and dep in providers_with_specs:
                dfs((dep, providers_with_specs[dep]), visited, stack)

        stack.append(api_str)

    visited: set[str] = set()
    stack: list[str] = []

    for api_str, providers in providers_with_specs.items():
        if api_str not in visited:
            dfs((api_str, providers), visited, stack)

    flattened = []
    for api_str in stack:
        for provider in providers_with_specs[api_str]:
            flattened.append((api_str, provider))

    return flattened


async def instantiate_provider(
    provider: ProviderWithSpec,
    deps: dict[Api, Any],
    inner_impls: dict[str, Any],
    dist_registry: DistributionRegistry,
    run_config: StackConfig,
    policy: list[AccessRule],
):
    """Instantiate a single provider, loading its module and verifying protocol compliance.

    Args:
        provider: The provider with its resolved spec.
        deps: Resolved API dependencies for this provider.
        inner_impls: Inner implementations for routing table providers.
        dist_registry: The distribution registry for resource management.
        run_config: The stack run configuration.
        policy: Access control policy rules.

    Returns:
        The instantiated provider implementation.
    """
    provider_spec = provider.spec
    if not hasattr(provider_spec, "module") or provider_spec.module is None:
        raise AttributeError(f"ProviderSpec of type {type(provider_spec)} does not have a 'module' attribute")

    logger.debug("Instantiating provider", provider_id=provider.provider_id, module=provider_spec.module)
    module = importlib.import_module(provider_spec.module)
    args = []
    if isinstance(provider_spec, RemoteProviderSpec):
        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider.config)

        method = "get_adapter_impl"
        args = [config, deps]

        if "policy" in inspect.signature(getattr(module, method)).parameters:
            args.append(policy)

    elif isinstance(provider_spec, AutoRoutedProviderSpec):
        method = "get_auto_router_impl"

        config = None
        args = [provider_spec.api, deps[provider_spec.routing_table_api], deps, run_config, policy]
    elif isinstance(provider_spec, RoutingTableProviderSpec):
        method = "get_routing_table_impl"

        config = None
        args = [provider_spec.api, inner_impls, deps, dist_registry, policy]
    else:
        method = "get_provider_impl"
        provider_config = provider.config.copy()

        # Inject vector_stores_config for providers that need it (introspection-based)
        config_type = instantiate_class_type(provider_spec.config_class)
        if hasattr(config_type, "__fields__") and "vector_stores_config" in config_type.__fields__:
            # Only inject if vector_stores is provided, otherwise let default_factory handle it
            if run_config.vector_stores is not None:
                provider_config["vector_stores_config"] = run_config.vector_stores

        config = config_type(**provider_config)
        args = [config, deps]
        if "policy" in inspect.signature(getattr(module, method)).parameters:
            args.append(policy)
    fn = getattr(module, method)
    impl = await fn(*args)
    impl.__provider_id__ = provider.provider_id
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config

    protocols = api_protocol_map_for_compliance_check(run_config)
    additional_protocols = additional_protocols_map()
    # TODO: check compliance for special tool groups
    # the impl should be for Api.tool_runtime, the name should be the special tool group, the protocol should be the special tool group protocol
    check_protocol_compliance(impl, protocols[provider_spec.api])
    if not isinstance(provider_spec, AutoRoutedProviderSpec) and provider_spec.api in additional_protocols:
        additional_api, _, _ = additional_protocols[provider_spec.api]
        check_protocol_compliance(impl, additional_api)

    return impl


def check_protocol_compliance(obj: Any, protocol: Any) -> None:
    """Verify that a provider implementation correctly implements all required protocol methods.

    Args:
        obj: The provider implementation to check.
        protocol: The protocol class defining required methods.

    Raises:
        ValueError: If the provider is missing required methods or has signature mismatches.
    """
    missing_methods = []

    mro = type(obj).__mro__
    for name, value in inspect.getmembers(protocol):
        if inspect.isfunction(value) and hasattr(value, "__webmethods__"):
            has_alpha_api = False
            for webmethod in value.__webmethods__:
                if webmethod.level == OGX_API_V1ALPHA:
                    has_alpha_api = True
                    break
            # if this API has multiple webmethods, and one of them is an alpha API, this API should be skipped when checking for missing or not callable routes
            if has_alpha_api:
                continue
            if not hasattr(obj, name):
                missing_methods.append((name, "missing"))
            elif not callable(getattr(obj, name)):
                missing_methods.append((name, "not_callable"))
            else:
                # Check if the method signatures are compatible
                obj_method = getattr(obj, name)
                proto_sig = inspect.signature(value)
                obj_sig = inspect.signature(obj_method)

                proto_params = set(proto_sig.parameters)
                proto_params.discard("self")
                obj_params = set(obj_sig.parameters)
                obj_params.discard("self")
                if not (proto_params <= obj_params):
                    logger.error(
                        "Method signature incompatible", method=name, proto_params=proto_params, obj_params=obj_params
                    )
                    missing_methods.append((name, "signature_mismatch"))
                else:
                    # Check if the method has a concrete implementation (not just a protocol stub)
                    # Find all classes in MRO that define this method
                    method_owners = [cls for cls in mro if name in cls.__dict__]

                    # Allow methods from mixins/parents, only reject if ONLY the protocol defines it
                    if len(method_owners) == 1 and method_owners[0].__name__ == protocol.__name__:
                        # Only reject if the method is ONLY defined in the protocol itself (abstract stub)
                        missing_methods.append((name, "not_actually_implemented"))

    if missing_methods:
        raise ValueError(
            f"Provider `{obj.__provider_id__} ({obj.__provider_spec__.api})` does not implement the following methods:\n{missing_methods}"
        )


async def resolve_remote_stack_impls(
    config: RemoteProviderConfig,
    apis: list[str],
) -> dict[Api, Any]:
    """Resolve provider implementations for a remote stack by creating API clients.

    Args:
        config: Remote provider configuration containing the connection URL.
        apis: List of API names to resolve.

    Returns:
        Dictionary mapping APIs to their remote client implementations.
    """
    protocols = api_protocol_map()
    additional_protocols = additional_protocols_map()

    impls = {}
    for api_str in apis:
        api = Api(api_str)
        impls[api] = await get_client_impl(
            protocols[api],
            config,
            {},
        )
        if api in additional_protocols:
            _, additional_protocol, additional_api = additional_protocols[api]
            impls[additional_api] = await get_client_impl(
                additional_protocol,
                config,
                {},
            )

    return impls
