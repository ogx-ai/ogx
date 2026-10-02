# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the shared integration-test helpers in ``tests/integration/helpers.py``.

``provider_from_model`` is used by every integration suite to decide which tests to
skip, so a silent regression there silently changes coverage.  These tests exercise it
against fakes instead of a live stack.
"""

import pytest

from tests.integration.helpers import provider_from_model


class FakeModel:
    def __init__(self, model_id: str, custom_metadata: dict | None):
        self.id = model_id
        self.custom_metadata = custom_metadata


class FakeProvider:
    def __init__(self, provider_id: str, provider_type: str):
        self.provider_id = provider_id
        self.provider_type = provider_type


class FakeModelsResource:
    def __init__(self, models: list[FakeModel], counter: list[int]):
        self._models = models
        self._counter = counter

    def list(self):
        self._counter[0] += 1
        return type("ModelList", (), {"data": list(self._models)})()


class FakeProvidersResource:
    def __init__(self, providers: list[FakeProvider]):
        self._providers = providers

    def list(self):
        return list(self._providers)


class FakeClient:
    """Minimal stand-in for the OGX client, counting ``models.list()`` round-trips."""

    def __init__(self, models: list[FakeModel], providers: list[FakeProvider]):
        self.list_calls = [0]
        self.models = FakeModelsResource(models, self.list_calls)
        self.providers = FakeProvidersResource(providers)


OPENAI_PROVIDER = FakeProvider("openai", "remote::openai")
VLLM_PROVIDER = FakeProvider("vllm", "remote::vllm")


def _client(*models: FakeModel) -> FakeClient:
    return FakeClient(list(models), [OPENAI_PROVIDER, VLLM_PROVIDER])


def test_resolves_by_model_id():
    client = _client(FakeModel("openai/gpt-4o", {"provider_id": "openai"}))

    assert provider_from_model(client, "openai/gpt-4o").provider_type == "remote::openai"


def test_resolves_by_provider_resource_id_alias():
    client = _client(
        FakeModel(
            "vllm/meta-llama/Llama-3.1-8B", {"provider_id": "vllm", "provider_resource_id": "meta-llama/Llama-3.1-8B"}
        )
    )

    assert provider_from_model(client, "meta-llama/Llama-3.1-8B").provider_type == "remote::vllm"


def test_alias_takes_precedence_over_model_id():
    """A model whose alias collides with another model's ID wins the lookup."""
    client = _client(
        FakeModel("shared-name", {"provider_id": "openai"}),
        FakeModel("vllm/shared-name", {"provider_id": "vllm", "provider_resource_id": "shared-name"}),
    )

    assert provider_from_model(client, "shared-name").provider_type == "remote::vllm"


def test_model_without_provider_resource_id_does_not_break_the_alias_pass():
    client = _client(
        FakeModel("openai/gpt-4o", {"provider_id": "openai"}),
        FakeModel("sentence-transformers/all-MiniLM-L6-v2", {"provider_id": "vllm"}),
        FakeModel("no-metadata-at-all", None),
    )

    assert provider_from_model(client, "openai/gpt-4o").provider_type == "remote::openai"


def test_models_are_listed_once_per_call():
    client = _client(FakeModel("openai/gpt-4o", {"provider_id": "openai"}))

    provider_from_model(client, "openai/gpt-4o")

    assert client.list_calls[0] == 1


def test_unknown_model_fails_with_a_descriptive_message():
    client = _client(FakeModel("openai/gpt-4o", {"provider_id": "openai"}))

    with pytest.raises(pytest.fail.Exception) as exc_info:
        provider_from_model(client, "vllm/meta-llama/Llama-3.1-8B")

    assert "Failed to resolve provider for model 'vllm/meta-llama/Llama-3.1-8B'" in str(exc_info.value)
    assert "not found among 1 listed models" in str(exc_info.value)


@pytest.mark.parametrize("custom_metadata", [None, {}, {"provider_resource_id": "gpt-4o"}])
def test_missing_provider_id_fails_with_a_descriptive_message(custom_metadata):
    client = _client(FakeModel("openai/gpt-4o", custom_metadata))

    with pytest.raises(pytest.fail.Exception) as exc_info:
        provider_from_model(client, "openai/gpt-4o")

    assert "no 'provider_id' in its custom_metadata" in str(exc_info.value)


def test_unregistered_provider_fails_with_a_descriptive_message():
    client = _client(FakeModel("openai/gpt-4o", {"provider_id": "ghost"}))

    with pytest.raises(pytest.fail.Exception) as exc_info:
        provider_from_model(client, "openai/gpt-4o")

    assert "provider 'ghost' is not registered" in str(exc_info.value)
