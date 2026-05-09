# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Unit tests for the routing tables

from unittest.mock import AsyncMock

import pytest

from ogx.core.datatypes import RegistryEntrySource
from ogx.core.routing_tables.models import ModelsRoutingTable
from ogx.core.routing_tables.shields import ShieldsRoutingTable
from ogx.core.routing_tables.toolgroups import ToolGroupsRoutingTable
from ogx_api import (
    URL,
    Api,
    GetShieldRequest,
    ListToolDefsResponse,
    ListToolsRequest,
    Model,
    ModelNotFoundError,
    ModelType,
    RegisterShieldRequest,
    Shield,
    ToolDef,
    ToolGroup,
    UnregisterShieldRequest,
)


class Impl:
    def __init__(self, api: Api):
        self.api = api

    @property
    def __provider_spec__(self):
        _provider_spec = AsyncMock()
        _provider_spec.api = self.api
        return _provider_spec


class InferenceImpl(Impl):
    def __init__(self):
        super().__init__(Api.inference)

    async def register_model(self, model: Model):
        return model

    async def unregister_model(self, model_id: str):
        return model_id

    async def should_refresh_models(self):
        return False

    async def list_models(self):
        return [
            Model(
                identifier="provider-model-1",
                provider_resource_id="provider-model-1",
                provider_id="test_provider",
                metadata={},
                model_type=ModelType.llm,
            ),
            Model(
                identifier="provider-model-2",
                provider_resource_id="provider-model-2",
                provider_id="test_provider",
                metadata={"embedding_dimension": 512},
                model_type=ModelType.embedding,
            ),
        ]

    async def shutdown(self):
        pass


class SafetyImpl(Impl):
    def __init__(self):
        super().__init__(Api.safety)

    async def register_shield(self, shield: Shield):
        return shield

    async def unregister_shield(self, shield_id: str):
        return shield_id


class ToolGroupsImpl(Impl):
    def __init__(self):
        super().__init__(Api.tool_runtime)

    async def register_toolgroup(self, toolgroup: ToolGroup):
        return toolgroup

    async def unregister_toolgroup(self, toolgroup_id: str):
        return toolgroup_id

    async def list_runtime_tools(self, toolgroup_id, mcp_endpoint, authorization=None):
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="test-tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {"test-param": {"type": "string", "description": "Test param"}},
                    },
                )
            ]
        )


async def test_models_routing_table(cached_disk_dist_registry):
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple models and verify listing
    await table.register_model(model_id="test-model", provider_id="test_provider")
    await table.register_model(model_id="test-model-2", provider_id="test_provider")

    models = await table.list_models()
    assert len(models.data) == 2
    model_ids = {m.identifier for m in models.data}
    assert "test_provider/test-model" in model_ids
    assert "test_provider/test-model-2" in model_ids

    # Test openai list models
    openai_models = await table.openai_list_models()
    assert len(openai_models.data) == 2
    openai_model_ids = {m.id for m in openai_models.data}
    assert "test_provider/test-model" in openai_model_ids
    assert "test_provider/test-model-2" in openai_model_ids

    # Verify custom_metadata is populated with OGX-specific data
    for openai_model in openai_models.data:
        assert openai_model.custom_metadata is not None
        assert "model_type" in openai_model.custom_metadata
        assert "provider_id" in openai_model.custom_metadata
        assert "provider_resource_id" in openai_model.custom_metadata
        assert openai_model.custom_metadata["provider_id"] == "test_provider"

    # Test get_object_by_identifier
    model = await table.get_object_by_identifier("model", "test_provider/test-model")
    assert model is not None
    assert model.identifier == "test_provider/test-model"

    # Test get_object_by_identifier on non-existent object
    non_existent = await table.get_object_by_identifier("model", "non-existent-model")
    assert non_existent is None

    # Test has_model
    assert await table.has_model("test_provider/test-model")
    assert await table.has_model("test_provider/test-model-2")
    assert not await table.has_model("non-existent-model")
    assert not await table.has_model("test_provider/non-existent-model")

    await table.unregister_model(model_id="test_provider/test-model")
    await table.unregister_model(model_id="test_provider/test-model-2")

    models = await table.list_models()
    assert len(models.data) == 0

    # Test openai list models
    openai_models = await table.openai_list_models()
    assert len(openai_models.data) == 0


async def test_shields_routing_table(cached_disk_dist_registry):
    table = ShieldsRoutingTable({"test_provider": SafetyImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple shields and verify listing
    await table.register_shield(RegisterShieldRequest(shield_id="test-shield", provider_id="test_provider"))
    await table.register_shield(RegisterShieldRequest(shield_id="test-shield-2", provider_id="test_provider"))
    shields = await table.list_shields()
    assert len(shields.data) == 2

    shield_ids = {s.identifier for s in shields.data}
    assert "test-shield" in shield_ids
    assert "test-shield-2" in shield_ids

    # Test get specific shield
    test_shield = await table.get_shield(GetShieldRequest(identifier="test-shield"))
    assert test_shield is not None
    assert test_shield.identifier == "test-shield"
    assert test_shield.provider_id == "test_provider"
    assert test_shield.provider_resource_id == "test-shield"
    assert test_shield.params == {}

    # Test get non-existent shield - should raise ValueError with specific message
    with pytest.raises(ValueError, match="Shield 'non-existent' not found"):
        await table.get_shield(GetShieldRequest(identifier="non-existent"))

    # Test unregistering shields
    await table.unregister_shield(UnregisterShieldRequest(identifier="test-shield"))
    shields = await table.list_shields()

    assert len(shields.data) == 1
    shield_ids = {s.identifier for s in shields.data}
    assert "test-shield" not in shield_ids
    assert "test-shield-2" in shield_ids

    # Unregister the remaining shield
    await table.unregister_shield(UnregisterShieldRequest(identifier="test-shield-2"))
    shields = await table.list_shields()
    assert len(shields.data) == 0

    # Test unregistering non-existent shield - should raise ValueError with specific message
    with pytest.raises(ValueError, match="Shield 'non-existent' not found"):
        await table.unregister_shield(UnregisterShieldRequest(identifier="non-existent"))


async def test_double_registration_models_positive(cached_disk_dist_registry):
    """Test that registering the same model twice with identical data succeeds."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a model
    await table.register_model(model_id="test-model", provider_id="test_provider", metadata={"param1": "value1"})

    # Register the exact same model again - should succeed (idempotent)
    await table.register_model(model_id="test-model", provider_id="test_provider", metadata={"param1": "value1"})

    # Verify only one model exists
    models = await table.list_models()
    assert len(models.data) == 1
    assert models.data[0].identifier == "test_provider/test-model"


async def test_double_registration_models_negative(cached_disk_dist_registry):
    """Test that registering the same model with conflicting data fails."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a model with specific metadata
    await table.register_model(model_id="test-model", provider_id="test_provider", metadata={"param1": "value1"})

    # Try to register the same model with conflicting metadata - should fail
    with pytest.raises(ValueError, match="conflicting field values"):
        await table.register_model(
            model_id="test-model", provider_id="test_provider", metadata={"param1": "different_value"}
        )


async def test_double_registration_different_providers(cached_disk_dist_registry):
    """Test that registering objects with same ID but different providers succeeds."""
    impl1 = InferenceImpl()
    impl2 = InferenceImpl()
    table = ModelsRoutingTable({"provider1": impl1, "provider2": impl2}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register same model ID with different providers - should succeed
    await table.register_model(model_id="shared-model", provider_id="provider1")
    await table.register_model(model_id="shared-model", provider_id="provider2")

    # Verify both models exist with different identifiers
    models = await table.list_models()
    assert len(models.data) == 2
    model_ids = {m.identifier for m in models.data}
    assert "provider1/shared-model" in model_ids
    assert "provider2/shared-model" in model_ids


async def test_openai_list_models_has_object_field(cached_disk_dist_registry):
    """Test that OpenAI list models response includes the object field."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    await table.register_model(model_id="test-model", provider_id="test_provider")

    openai_models = await table.openai_list_models()
    assert openai_models.object == "list"
    assert len(openai_models.data) == 1


async def test_model_has_openai_compatible_fields(cached_disk_dist_registry):
    """Test that Model includes OpenAI-compatible fields (id, object, created, owned_by)."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    await table.register_model(model_id="test-model", provider_id="test_provider")

    model = await table.get_model("test_provider/test-model")
    assert model.id == "test_provider/test-model"
    assert model.object == "model"
    assert isinstance(model.created, int)
    assert model.owned_by == "ogx"

    # Verify the fields appear in serialized output
    data = model.model_dump()
    assert "id" in data
    assert "object" in data
    assert "created" in data
    assert "owned_by" in data
    assert data["id"] == model.identifier


async def test_tool_groups_routing_table(cached_disk_dist_registry):
    table = ToolGroupsRoutingTable({"test_provider": ToolGroupsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple tool groups and verify listing
    await table.register_tool_group(
        toolgroup_id="test-toolgroup",
        provider_id="test_provider",
    )
    tool_groups = await table.list_tool_groups()

    assert len(tool_groups.data) == 1
    tool_group_ids = {tg.identifier for tg in tool_groups.data}
    assert "test-toolgroup" in tool_group_ids

    await table.unregister_toolgroup(toolgroup_id="test-toolgroup")
    tool_groups = await table.list_tool_groups()
    assert len(tool_groups.data) == 0


async def test_models_alias_registration_and_lookup(cached_disk_dist_registry):
    """Test alias registration (model_id != provider_model_id) and lookup behavior."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register model with alias (model_id different from provider_model_id)
    # The identifier uses model_id, while provider_resource_id stores the actual provider model
    await table.register_model(
        model_id="my-alias", provider_model_id="actual-provider-model", provider_id="test_provider"
    )

    # Verify the model was registered with model_id as identifier
    models = await table.list_models()
    assert len(models.data) == 1
    model = models.data[0]
    assert model.identifier == "test_provider/my-alias"
    assert model.provider_resource_id == "actual-provider-model"

    # Test lookup by unprefixed alias fails
    with pytest.raises(ModelNotFoundError, match="Model 'my-alias' not found"):
        await table.get_model("my-alias")

    retrieved_model = await table.get_model("test_provider/my-alias")
    assert retrieved_model.identifier == "test_provider/my-alias"
    assert retrieved_model.provider_resource_id == "actual-provider-model"


async def test_models_multi_provider_disambiguation(cached_disk_dist_registry):
    """Test registration and lookup with multiple providers having same provider_model_id."""
    table = ModelsRoutingTable(
        {"provider1": InferenceImpl(), "provider2": InferenceImpl()}, cached_disk_dist_registry, {}
    )
    await table.initialize()

    # Register same provider_model_id on both providers (no aliases)
    await table.register_model(model_id="common-model", provider_id="provider1")
    await table.register_model(model_id="common-model", provider_id="provider2")

    # Verify both models get namespaced identifiers
    models = await table.list_models()
    assert len(models.data) == 2
    identifiers = {m.identifier for m in models.data}
    assert identifiers == {"provider1/common-model", "provider2/common-model"}

    # Test lookup by full namespaced identifier works
    model1 = await table.get_model("provider1/common-model")
    assert model1.provider_id == "provider1"
    assert model1.provider_resource_id == "common-model"

    model2 = await table.get_model("provider2/common-model")
    assert model2.provider_id == "provider2"
    assert model2.provider_resource_id == "common-model"

    # Test lookup by unscoped provider_model_id fails with multiple providers error
    with pytest.raises(ModelNotFoundError, match="Model 'common-model' not found"):
        await table.get_model("common-model")


async def test_models_fallback_lookup_behavior(cached_disk_dist_registry):
    """Test two-stage lookup: direct identifier hit vs fallback to provider_resource_id."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register model without alias (gets namespaced identifier)
    await table.register_model(model_id="test-model", provider_id="test_provider")

    # Verify namespaced identifier was created
    models = await table.list_models()
    assert len(models.data) == 1
    model = models.data[0]
    assert model.identifier == "test_provider/test-model"
    assert model.provider_resource_id == "test-model"

    # Test lookup by full namespaced identifier (direct hit via get_object_by_identifier)
    retrieved_model = await table.get_model("test_provider/test-model")
    assert retrieved_model.identifier == "test_provider/test-model"

    # Test lookup by unscoped provider_model_id (fallback via iteration)
    with pytest.raises(ModelNotFoundError, match="Model 'test-model' not found"):
        await table.get_model("test-model")

    # Test lookup of non-existent model fails
    with pytest.raises(ModelNotFoundError, match="Model 'non-existent' not found"):
        await table.get_model("non-existent")


async def test_models_source_tracking_default(cached_disk_dist_registry):
    """Test that models registered via register_model get default source."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register model via register_model (should get default source)
    await table.register_model(model_id="user-model", provider_id="test_provider")

    models = await table.list_models()
    assert len(models.data) == 1
    model = models.data[0]
    assert model.source == RegistryEntrySource.via_register_api
    assert model.identifier == "test_provider/user-model"

    # Cleanup
    await table.shutdown()


async def test_models_source_tracking_provider(cached_disk_dist_registry):
    """Test that models registered via update_registered_models get provider source."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Simulate provider refresh by calling update_registered_models
    provider_models = [
        Model(
            identifier="provider-model-1",
            provider_resource_id="provider-model-1",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
        Model(
            identifier="provider-model-2",
            provider_resource_id="provider-model-2",
            provider_id="test_provider",
            metadata={"embedding_dimension": 512},
            model_type=ModelType.embedding,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models)

    models = await table.list_models()
    assert len(models.data) == 2

    # All models should have provider source
    for model in models.data:
        assert model.source == RegistryEntrySource.listed_from_provider
        assert model.provider_id == "test_provider"

    # Cleanup
    await table.shutdown()


async def test_models_source_interaction_preserves_default(cached_disk_dist_registry):
    """Test that provider refresh preserves user-registered models with default source."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # First register a user model with same provider_resource_id as provider will later provide
    await table.register_model(
        model_id="my-custom-alias", provider_model_id="provider-model-1", provider_id="test_provider"
    )

    # Verify user model is registered with default source
    models = await table.list_models()
    assert len(models.data) == 1
    user_model = models.data[0]
    assert user_model.source == RegistryEntrySource.via_register_api
    assert user_model.identifier == "test_provider/my-custom-alias"
    assert user_model.provider_resource_id == "provider-model-1"

    # Now simulate provider refresh
    provider_models = [
        Model(
            identifier="provider-model-1",
            provider_resource_id="provider-model-1",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
        Model(
            identifier="different-model",
            provider_resource_id="different-model",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models)

    # Verify user model with alias is preserved, provider-listed model is also registered,
    # and provider added new model (3 total: user alias + 2 provider models)
    models = await table.list_models()
    assert len(models.data) == 3

    # Find the user alias, provider-listed model, and different model
    user_alias = next((m for m in models.data if m.identifier == "test_provider/my-custom-alias"), None)
    provider_listed = next((m for m in models.data if m.identifier == "test_provider/provider-model-1"), None)
    different_model = next((m for m in models.data if m.identifier == "test_provider/different-model"), None)

    # User-registered alias should be preserved
    assert user_alias is not None
    assert user_alias.source == RegistryEntrySource.via_register_api
    assert user_alias.provider_resource_id == "provider-model-1"

    # Provider-listed model with same provider_resource_id should also exist
    # (allows both alias and canonical name to coexist)
    assert provider_listed is not None
    assert provider_listed.source == RegistryEntrySource.listed_from_provider
    assert provider_listed.provider_resource_id == "provider-model-1"

    # Different provider model should be added
    assert different_model is not None
    assert different_model.source == RegistryEntrySource.listed_from_provider
    assert different_model.provider_resource_id == "different-model"

    # Cleanup
    await table.shutdown()


async def test_models_source_interaction_cleanup_provider_models(cached_disk_dist_registry):
    """Test that provider refresh removes old provider models but keeps default ones."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a user model
    await table.register_model(model_id="user-model", provider_id="test_provider")

    # Add some provider models
    provider_models_v1 = [
        Model(
            identifier="provider-model-old",
            provider_resource_id="provider-model-old",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models_v1)

    # Verify we have both user and provider models
    models = await table.list_models()
    assert len(models.data) == 2

    # Now update with new provider models (should remove old provider models)
    provider_models_v2 = [
        Model(
            identifier="provider-model-new",
            provider_resource_id="provider-model-new",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models_v2)

    # Should have user model + new provider model, old provider model gone
    models = await table.list_models()
    assert len(models.data) == 2

    identifiers = {m.identifier for m in models.data}
    assert "test_provider/user-model" in identifiers  # User model preserved
    assert "test_provider/provider-model-new" in identifiers  # New provider model (uses provider's identifier)
    assert "test_provider/provider-model-old" not in identifiers  # Old provider model removed

    # Verify sources are correct
    user_model = next((m for m in models.data if m.identifier == "test_provider/user-model"), None)
    provider_model = next((m for m in models.data if m.identifier == "test_provider/provider-model-new"), None)

    assert user_model.source == RegistryEntrySource.via_register_api
    assert provider_model.source == RegistryEntrySource.listed_from_provider

    # Cleanup
    await table.shutdown()


async def test_tool_groups_routing_table_exception_handling(cached_disk_dist_registry):
    """Test that the tool group routing table handles exceptions when listing tools, like if an MCP server is unreachable."""

    exception_throwing_tool_groups_impl = ToolGroupsImpl()
    exception_throwing_tool_groups_impl.list_runtime_tools = AsyncMock(side_effect=Exception("Test exception"))

    table = ToolGroupsRoutingTable(
        {"test_provider": exception_throwing_tool_groups_impl}, cached_disk_dist_registry, {}
    )
    await table.initialize()

    await table.register_tool_group(
        toolgroup_id="test-toolgroup-exceptions",
        provider_id="test_provider",
        mcp_endpoint=URL(uri="http://localhost:8479/foo/bar"),
    )

    tools = await table.list_tools(ListToolsRequest(toolgroup_id="test-toolgroup-exceptions"))

    assert len(tools.data) == 0
