# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

from ogx.core.datatypes import StackConfig
from ogx.core.server.server import ALWAYS_SERVED_APIS, RESPONSES_IMPLIED_APIS, apis_to_serve
from ogx_api import Api


def make_impls(*apis: Api) -> dict[Api, object]:
    return {api: Mock() for api in apis}


def test_absent_apis_list_serves_every_impl():
    config = StackConfig(distro_name="test", providers={})
    impls = make_impls(Api.inference, Api.responses, Api.conversations)

    served = apis_to_serve(config, impls)

    assert {"inference", "responses", "conversations"} <= served
    assert set(ALWAYS_SERVED_APIS) <= served


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

    assert set(ALWAYS_SERVED_APIS) <= apis_to_serve(config, make_impls(Api.inference))


def test_routing_table_api_follows_its_router_api():
    impls = make_impls(Api.inference, Api.models)

    with_inference = apis_to_serve(StackConfig(distro_name="test", apis=["inference"], providers={}), impls)
    without_inference = apis_to_serve(StackConfig(distro_name="test", apis=["files"], providers={}), impls)

    assert "models" in with_inference
    assert "models" not in without_inference
