# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

#
# ModelRegistryHelper provides mixin functionality for registering and
# unregistering models. It maintains a mapping of model ID / aliases to
# provider model IDs.
#
# Test cases -
#  - Looking up an alias that does not exist should return None.
#  - Registering a model + provider ID should add the model to the registry. If
#    provider ID is known or an alias for a provider ID.
#  - Registering an existing model should return an error. Unless it's a
#    dulicate entry.
#  - Unregistering a model should remove it from the registry.
#  - Unregistering a model that does not exist should return an error.
#  - Supported model ID and their aliases are registered during initialization.
#    Only aliases are added afterwards.
#
# Questions -
#  - Should we be allowed to register models w/o provider model IDs? No.
#    According to POST /v1/models, required params are
#      - identifier
#      - provider_resource_id
#      - provider_id
#      - type
#      - metadata
#      - model_type
#
#  TODO: llama_model functionality
#

import pytest

from ogx.providers.utils.inference.model_registry import ModelRegistryHelper, ProviderModelEntry
from ogx_api import Model, UnsupportedModelError


@pytest.fixture
def known_model() -> Model:
    return Model(
        provider_id="provider",
        identifier="known-model",
        provider_resource_id="known-provider-id",
    )


@pytest.fixture
def known_model2() -> Model:
    return Model(
        provider_id="provider",
        identifier="known-model2",
        provider_resource_id="known-provider-id2",
    )


@pytest.fixture
def known_provider_model(known_model: Model) -> ProviderModelEntry:
    return ProviderModelEntry(
        provider_model_id=known_model.provider_resource_id,
        aliases=[known_model.model_id],
    )


@pytest.fixture
def known_provider_model2(known_model2: Model) -> ProviderModelEntry:
    return ProviderModelEntry(
        provider_model_id=known_model2.provider_resource_id,
        # aliases=[],
    )


@pytest.fixture
def unknown_model() -> Model:
    return Model(
        provider_id="provider",
        identifier="unknown-model",
        provider_resource_id="unknown-provider-id",
    )


@pytest.fixture
def helper(known_provider_model: ProviderModelEntry, known_provider_model2: ProviderModelEntry) -> ModelRegistryHelper:
    return ModelRegistryHelper(model_entries=[known_provider_model, known_provider_model2])


class MockModelRegistryHelperWithDynamicModels(ModelRegistryHelper):
    """Test helper that simulates a provider with dynamically available models."""

    def __init__(self, model_entries: list[ProviderModelEntry], available_models: list[str]):
        super().__init__(model_entries=model_entries)
        self._available_models = available_models

    async def check_model_availability(self, model: str) -> bool:
        return model in self._available_models


@pytest.fixture
def dynamic_model() -> Model:
    """A model that's not in static config but available dynamically."""
    return Model(
        provider_id="provider",
        identifier="dynamic-model",
        provider_resource_id="dynamic-provider-id",
    )


@pytest.fixture
def helper_with_dynamic_models(
    known_provider_model: ProviderModelEntry, known_provider_model2: ProviderModelEntry, dynamic_model: Model
) -> MockModelRegistryHelperWithDynamicModels:
    """Helper that includes dynamically available models."""
    return MockModelRegistryHelperWithDynamicModels(
        [known_provider_model, known_provider_model2], [dynamic_model.provider_resource_id]
    )


async def test_lookup_unknown_model(helper: ModelRegistryHelper, unknown_model: Model) -> None:
    assert helper.get_provider_model_id(unknown_model.model_id) is None


async def test_register_unknown_provider_model(helper: ModelRegistryHelper, unknown_model: Model) -> None:
    with pytest.raises(UnsupportedModelError):
        await helper.register_model(unknown_model)


async def test_register_model(helper: ModelRegistryHelper, known_model: Model) -> None:
    model = Model(
        provider_id=known_model.provider_id,
        identifier="new-model",
        provider_resource_id=known_model.provider_resource_id,
    )
    assert helper.get_provider_model_id(model.model_id) is None
    await helper.register_model(model)
    assert helper.get_provider_model_id(model.model_id) == model.provider_resource_id


async def test_register_model_from_alias(helper: ModelRegistryHelper, known_model: Model) -> None:
    model = Model(
        provider_id=known_model.provider_id,
        identifier="new-model",
        provider_resource_id=known_model.model_id,  # use known model's id as an alias for the supported model id
    )
    assert helper.get_provider_model_id(model.model_id) is None
    await helper.register_model(model)
    assert helper.get_provider_model_id(model.model_id) == known_model.provider_resource_id


async def test_register_model_existing(helper: ModelRegistryHelper, known_model: Model) -> None:
    await helper.register_model(known_model)
    assert helper.get_provider_model_id(known_model.model_id) == known_model.provider_resource_id


async def test_register_model_existing_different(
    helper: ModelRegistryHelper, known_model: Model, known_model2: Model
) -> None:
    known_model.provider_resource_id = known_model2.provider_resource_id
    with pytest.raises(ValueError):
        await helper.register_model(known_model)


# TODO: unregister_model functionality was removed/disabled by https://github.com/ogx-ai/ogx/pull/2916
# async def test_unregister_model(helper: ModelRegistryHelper, known_model: Model) -> None:
#     await helper.register_model(known_model)  # duplicate entry
#     assert helper.get_provider_model_id(known_model.model_id) == known_model.provider_model_id
#     await helper.unregister_model(known_model.model_id)
#     assert helper.get_provider_model_id(known_model.model_id) is None


# TODO: unregister_model functionality was removed/disabled by https://github.com/ogx-ai/ogx/pull/2916
# async def test_unregister_unknown_model(helper: ModelRegistryHelper, unknown_model: Model) -> None:
#     with pytest.raises(ValueError):
#         await helper.unregister_model(unknown_model.model_id)


async def test_register_model_during_init(helper: ModelRegistryHelper, known_model: Model) -> None:
    assert helper.get_provider_model_id(known_model.provider_resource_id) == known_model.provider_model_id


# TODO: unregister_model functionality was removed/disabled by https://github.com/ogx-ai/ogx/pull/2916
# async def test_unregister_model_during_init(helper: ModelRegistryHelper, known_model: Model) -> None:
#     assert helper.get_provider_model_id(known_model.provider_resource_id) == known_model.provider_model_id
#     await helper.unregister_model(known_model.provider_resource_id)
#     assert helper.get_provider_model_id(known_model.provider_resource_id) is None


async def test_register_model_from_check_model_availability(
    helper_with_dynamic_models: MockModelRegistryHelperWithDynamicModels, dynamic_model: Model
) -> None:
    """Test that models returned by check_model_availability can be registered."""
    # Verify the model is not in static config
    assert helper_with_dynamic_models.get_provider_model_id(dynamic_model.provider_resource_id) is None

    # But it should be available via check_model_availability
    is_available = await helper_with_dynamic_models.check_model_availability(dynamic_model.provider_resource_id)
    assert is_available

    # Registration should succeed
    registered_model = await helper_with_dynamic_models.register_model(dynamic_model)
    assert registered_model == dynamic_model

    # Model should now be registered and accessible
    assert (
        helper_with_dynamic_models.get_provider_model_id(dynamic_model.model_id) == dynamic_model.provider_resource_id
    )


async def test_register_model_not_in_static_or_dynamic(
    helper_with_dynamic_models: MockModelRegistryHelperWithDynamicModels, unknown_model: Model
) -> None:
    """Test that models not in static config or dynamic models are rejected."""
    # Verify the model is not in static config
    assert helper_with_dynamic_models.get_provider_model_id(unknown_model.provider_resource_id) is None

    # And not available via check_model_availability
    is_available = await helper_with_dynamic_models.check_model_availability(unknown_model.provider_resource_id)
    assert not is_available

    # Registration should fail with comprehensive error message
    with pytest.raises(Exception) as exc_info:  # UnsupportedModelError
        await helper_with_dynamic_models.register_model(unknown_model)

    # Error should include static models and "..." for dynamic models
    error_str = str(exc_info.value)
    assert "..." in error_str  # "..." should be in error message


async def test_register_alias_for_dynamic_model(
    helper_with_dynamic_models: MockModelRegistryHelperWithDynamicModels, dynamic_model: Model
) -> None:
    """Test that we can register an alias that maps to a dynamically available model."""
    # Create a model with a different identifier but same provider_resource_id
    alias_model = Model(
        provider_id=dynamic_model.provider_id,
        identifier="dynamic-model-alias",
        provider_resource_id=dynamic_model.provider_resource_id,
    )

    # Registration should succeed since the provider_resource_id is available dynamically
    registered_model = await helper_with_dynamic_models.register_model(alias_model)
    assert registered_model == alias_model

    # Both the original provider_resource_id and the new alias should work
    assert helper_with_dynamic_models.get_provider_model_id(alias_model.model_id) == dynamic_model.provider_resource_id
