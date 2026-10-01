# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, Mock

from ogx.core.datatypes import StackConfig
from ogx.core.server.fastapi_router_registry import collect_api_routes
from ogx.core.server.server import (
    ALWAYS_SERVED_APIS,
    RESPONSES_IMPLIED_APIS,
    StackApp,
    apis_to_serve,
    lifespan,
)
from ogx_api import Api

# Served regardless of `apis:`, so every expectation below is stated relative to them.
UNGATED_APIS = set(ALWAYS_SERVED_APIS)


def make_impls(*apis: Api) -> dict[Api, object]:
    return {api: Mock() for api in apis}


def test_absent_apis_list_serves_every_impl():
    config = StackConfig(distro_name="test", providers={})
    impls = make_impls(Api.inference, Api.responses, Api.conversations, Api.prompts)

    assert (
        apis_to_serve(config, impls) == {"inference", "responses", "conversations", "prompts", "models"} | UNGATED_APIS
    )


def test_empty_apis_list_serves_no_provider_backed_api():
    config = StackConfig(distro_name="test", apis=[], providers={})
    impls = make_impls(Api.inference, Api.responses)

    assert apis_to_serve(config, impls) == UNGATED_APIS


def test_explicit_apis_list_serves_only_what_it_names():
    config = StackConfig(distro_name="test", apis=["responses"], providers={})

    served = apis_to_serve(config, make_impls(Api.inference, Api.responses))

    assert "responses" in served
    assert "inference" not in served


def test_serving_responses_implies_conversations_and_prompts():
    """A responses deployment gets the built-in APIs its clients expect, unlisted."""
    config = StackConfig(distro_name="test", apis=["responses"], providers={})

    served = apis_to_serve(config, make_impls(Api.responses, Api.conversations, Api.prompts))

    assert set(RESPONSES_IMPLIED_APIS) <= served


def test_conversations_is_served_when_listed_without_responses():
    """Explicitly opting in still works for a deployment that does not serve responses."""
    config = StackConfig(distro_name="test", apis=["conversations"], providers={})

    served = apis_to_serve(config, make_impls(Api.inference, Api.conversations))

    assert "conversations" in served
    assert "responses" not in served
    assert "prompts" not in served


def test_responses_less_deployment_serves_neither_implied_api():
    """The gateway topology: dropping `responses` from `apis:` turns both off with it."""
    config = StackConfig(distro_name="test", apis=["inference"], providers={})

    served = apis_to_serve(config, make_impls(Api.inference, Api.responses, Api.conversations, Api.prompts))

    assert "conversations" not in served
    assert "prompts" not in served
    assert "responses" not in served


def test_administration_apis_are_served_even_when_omitted():
    config = StackConfig(distro_name="test", apis=["inference"], providers={})

    assert UNGATED_APIS <= apis_to_serve(config, make_impls(Api.inference))


def test_routing_table_api_follows_its_router_api():
    impls = make_impls(Api.inference, Api.models)

    with_inference = apis_to_serve(StackConfig(distro_name="test", apis=["inference"], providers={}), impls)
    without_inference = apis_to_serve(StackConfig(distro_name="test", apis=["files"], providers={}), impls)

    assert "models" in with_inference
    assert "models" not in without_inference


def _lifespan_app(config: StackConfig) -> StackApp:
    """Build a StackApp whose stack is stubbed but whose real lifespan registers real routers."""
    app = StackApp(config)
    app.stack = Mock()
    app.stack.run_config = config
    # Every API lifespan might register must have an impl, including the ungated ones.
    app.stack.impls = make_impls(
        Api.admin,
        Api.conversations,
        Api.inference,
        Api.inspect,
        Api.models,
        Api.prompts,
        Api.providers,
        Api.responses,
    )
    app.stack.initialize = AsyncMock()
    app.stack.shutdown = AsyncMock()
    return app


async def _registered_paths(app: StackApp) -> set[str]:
    async with lifespan(app):
        # collect_api_routes resolves included routers, which newer FastAPI
        # versions keep nested in app.routes instead of flattening.
        return {route.path for route in collect_api_routes(app.routes)}


async def test_lifespan_empty_apis_list_registers_no_provider_routers():
    app = _lifespan_app(StackConfig(distro_name="test", apis=[], providers={}))

    paths = await _registered_paths(app)

    assert not any(path.startswith("/v1/responses") for path in paths)
    assert not any(path.startswith("/v1/inference") for path in paths)
    assert "/v1/conversations" not in paths


async def test_lifespan_absent_apis_list_registers_provider_routers():
    app = _lifespan_app(StackConfig(distro_name="test", providers={}))

    paths = await _registered_paths(app)

    assert any(path.startswith("/v1/responses") for path in paths)
    assert "/v1/conversations" in paths
