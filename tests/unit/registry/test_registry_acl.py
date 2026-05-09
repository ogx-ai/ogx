# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.core.datatypes import ModelWithOwner, User
from ogx.core.store.registry import CachedDiskDistributionRegistry
from ogx_api import ModelType


async def test_registry_cache_with_acl(cached_disk_dist_registry):
    model = ModelWithOwner(
        identifier="model-acl",
        provider_id="test-provider",
        provider_resource_id="model-acl-resource",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"], "teams": ["ai-team"]}),
    )

    success = await cached_disk_dist_registry.register(model)
    assert success

    cached_model = cached_disk_dist_registry.get_cached("model", "model-acl")
    assert cached_model is not None
    assert cached_model.identifier == "model-acl"
    assert cached_model.owner.principal == "testuser"
    assert cached_model.owner.attributes["roles"] == ["admin"]
    assert cached_model.owner.attributes["teams"] == ["ai-team"]

    fetched_model = await cached_disk_dist_registry.get("model", "model-acl")
    assert fetched_model is not None
    assert fetched_model.identifier == "model-acl"
    assert fetched_model.owner.attributes["roles"] == ["admin"]

    new_registry = CachedDiskDistributionRegistry(cached_disk_dist_registry.kvstore)
    await new_registry.initialize()

    new_model = await new_registry.get("model", "model-acl")
    assert new_model is not None
    assert new_model.identifier == "model-acl"
    assert new_model.owner.principal == "testuser"
    assert new_model.owner.attributes["roles"] == ["admin"]
    assert new_model.owner.attributes["teams"] == ["ai-team"]


async def test_registry_empty_acl(cached_disk_dist_registry):
    model = ModelWithOwner(
        identifier="model-empty-acl",
        provider_id="test-provider",
        provider_resource_id="model-resource",
        model_type=ModelType.llm,
        owner=User("testuser", None),
    )

    await cached_disk_dist_registry.register(model)

    cached_model = cached_disk_dist_registry.get_cached("model", "model-empty-acl")
    assert cached_model is not None
    assert cached_model.owner is not None
    assert cached_model.owner.attributes is None

    all_models = await cached_disk_dist_registry.get_all()
    assert len(all_models) == 1

    model = ModelWithOwner(
        identifier="model-no-acl",
        provider_id="test-provider",
        provider_resource_id="model-resource-2",
        model_type=ModelType.llm,
    )

    await cached_disk_dist_registry.register(model)

    cached_model = cached_disk_dist_registry.get_cached("model", "model-no-acl")
    assert cached_model is not None
    assert cached_model.owner is None

    all_models = await cached_disk_dist_registry.get_all()
    assert len(all_models) == 2


async def test_registry_serialization(cached_disk_dist_registry):
    attributes = {
        "roles": ["admin", "researcher"],
        "teams": ["ai-team", "ml-team"],
        "projects": ["project-a", "project-b"],
        "namespaces": ["prod", "staging"],
    }

    model = ModelWithOwner(
        identifier="model-serialize",
        provider_id="test-provider",
        provider_resource_id="model-resource",
        model_type=ModelType.llm,
        owner=User("bob", attributes),
    )

    await cached_disk_dist_registry.register(model)

    new_registry = CachedDiskDistributionRegistry(cached_disk_dist_registry.kvstore)
    await new_registry.initialize()

    loaded_model = await new_registry.get("model", "model-serialize")
    assert loaded_model is not None

    assert loaded_model.owner.attributes["roles"] == ["admin", "researcher"]
    assert loaded_model.owner.attributes["teams"] == ["ai-team", "ml-team"]
    assert loaded_model.owner.attributes["projects"] == ["project-a", "project-b"]
    assert loaded_model.owner.attributes["namespaces"] == ["prod", "staging"]
