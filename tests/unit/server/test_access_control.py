# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from ogx.core.access_control.access_control import AccessDeniedError, is_action_allowed
from ogx.core.datatypes import AccessRule, ModelWithOwner, User
from ogx.core.routers.inference import InferenceRouter
from ogx.core.routing_tables.models import ModelsRoutingTable
from ogx_api import Api, Model, ModelNotFoundError, ModelType


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def _return_model(model):
    return model


@pytest.fixture
async def test_setup(cached_disk_dist_registry):
    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=cached_disk_dist_registry,
        policy={},
    )
    yield cached_disk_dist_registry, routing_table


@patch("ogx.core.routing_tables.common.get_authenticated_user")
async def test_access_control_with_cache(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-public",
        provider_id="test_provider",
        provider_resource_id="model-public",
        model_type=ModelType.llm,
    )
    model_admin_only = ModelWithOwner(
        identifier="model-admin",
        provider_id="test_provider",
        provider_resource_id="model-admin",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"]}),
    )
    model_data_scientist = ModelWithOwner(
        identifier="model-data-scientist",
        provider_id="test_provider",
        provider_resource_id="model-data-scientist",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["data-scientist", "researcher"], "teams": ["ml-team"]}),
    )
    await registry.register(model_public)
    await registry.register(model_admin_only)
    await registry.register(model_data_scientist)

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["admin"], "teams": ["management"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2

    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-admin")
    assert model.identifier == "model-admin"
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-data-scientist")

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["user"], "teams": ["other-team"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public"
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-admin")
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-data-scientist")

    mock_get_authenticated_user.return_value = User("test-user", {"roles": ["data-scientist"], "teams": ["ml-team"]})
    all_models = await routing_table.list_models()
    assert len(all_models.data) == 2
    model_ids = [m.identifier for m in all_models.data]
    assert "model-public" in model_ids
    assert "model-data-scientist" in model_ids
    assert "model-admin" not in model_ids
    model = await routing_table.get_model("model-public")
    assert model.identifier == "model-public"
    model = await routing_table.get_model("model-data-scientist")
    assert model.identifier == "model-data-scientist"
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-admin")


@patch("ogx.core.routing_tables.common.get_authenticated_user")
async def test_access_control_and_updates(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-updates",
        provider_id="test_provider",
        provider_resource_id="model-updates",
        model_type=ModelType.llm,
    )
    await registry.register(model_public)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["user"],
        },
    )
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"
    model_public.owner = User("testuser", {"roles": ["admin"]})
    await registry.update(model_public)
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["user"],
        },
    )
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-updates")
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["admin"],
        },
    )
    model = await routing_table.get_model("model-updates")
    assert model.identifier == "model-updates"


@patch("ogx.core.routing_tables.common.get_authenticated_user")
async def test_access_control_empty_attributes(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model = ModelWithOwner(
        identifier="model-empty-attrs",
        provider_id="test_provider",
        provider_resource_id="model-empty-attrs",
        model_type=ModelType.llm,
        owner=User("testuser", {}),
    )
    await registry.register(model)
    mock_get_authenticated_user.return_value = User(
        "differentuser",
        {
            "roles": [],
        },
    )
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-empty-attrs")
    all_models = await routing_table.list_models()
    model_ids = [m.identifier for m in all_models.data]
    assert "model-empty-attrs" not in model_ids


@patch("ogx.core.routing_tables.common.get_authenticated_user")
async def test_no_user_attributes(mock_get_authenticated_user, test_setup):
    registry, routing_table = test_setup
    model_public = ModelWithOwner(
        identifier="model-public-2",
        provider_id="test_provider",
        provider_resource_id="model-public-2",
        model_type=ModelType.llm,
    )
    model_restricted = ModelWithOwner(
        identifier="model-restricted",
        provider_id="test_provider",
        provider_resource_id="model-restricted",
        model_type=ModelType.llm,
        owner=User("testuser", {"roles": ["admin"]}),
    )
    await registry.register(model_public)
    await registry.register(model_restricted)
    mock_get_authenticated_user.return_value = User("test-user", None)
    model = await routing_table.get_model("model-public-2")
    assert model.identifier == "model-public-2"

    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("model-restricted")

    all_models = await routing_table.list_models()
    assert len(all_models.data) == 1
    assert all_models.data[0].identifier == "model-public-2"


@patch("ogx.core.routing_tables.common.get_authenticated_user")
async def test_automatic_access_attributes(mock_get_authenticated_user, test_setup):
    """Test that newly created resources inherit access attributes from their creator."""
    registry, routing_table = test_setup

    # Set creator's attributes
    creator_attributes = {"roles": ["data-scientist"], "teams": ["ml-team"], "projects": ["llama-3"]}
    mock_get_authenticated_user.return_value = User("test-user", creator_attributes)

    # Create model without explicit access attributes
    model = ModelWithOwner(
        identifier="auto-access-model",
        provider_id="test_provider",
        provider_resource_id="auto-access-model",
        model_type=ModelType.llm,
    )
    await routing_table.register_object(model)

    # Verify the model got creator's attributes
    registered_model = await routing_table.get_model("auto-access-model")
    assert registered_model.owner is not None
    assert registered_model.owner.attributes is not None
    assert registered_model.owner.attributes["roles"] == ["data-scientist"]
    assert registered_model.owner.attributes["teams"] == ["ml-team"]
    assert registered_model.owner.attributes["projects"] == ["llama-3"]

    # Verify another user without matching attributes can't access it
    mock_get_authenticated_user.return_value = User("test-user2", {"roles": ["engineer"], "teams": ["infra-team"]})
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("auto-access-model")

    # But a user with matching attributes can
    mock_get_authenticated_user.return_value = User(
        "test-user",
        {
            "roles": ["data-scientist", "engineer"],
            "teams": ["ml-team", "platform-team"],
            "projects": ["llama-3"],
        },
    )
    model = await routing_table.get_model("auto-access-model")
    assert model.identifier == "auto-access-model"


@pytest.fixture
async def test_setup_with_access_policy(cached_disk_dist_registry):
    mock_inference = Mock()
    mock_inference.__provider_spec__ = MagicMock()
    mock_inference.__provider_spec__.api = Api.inference
    mock_inference.register_model = AsyncMock(side_effect=_return_model)
    mock_inference.unregister_model = AsyncMock(side_effect=_return_model)

    config = """
                - permit:
                    principal: user-1
                    actions: [create, read, delete]
                    description: user-1 has full access to all models
                - permit:
                    principal: user-2
                    actions: [read]
                    resource: model::test_provider/model-1
                    description: user-2 has read access to model-1 only
                - permit:
                    principal: user-3
                    actions: [read]
                    resource: model::test_provider/model-2
                    description: user-3 has read access to model-2 only
                - forbid:
                    actions: [create, read, delete]
             """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    routing_table = ModelsRoutingTable(
        impls_by_provider_id={"test_provider": mock_inference},
        dist_registry=cached_disk_dist_registry,
        policy=policy,
    )
    yield routing_table


@patch("ogx.core.routing_tables.common.get_authenticated_user")
async def test_access_policy(mock_get_authenticated_user, test_setup_with_access_policy):
    routing_table = test_setup_with_access_policy
    mock_get_authenticated_user.return_value = User(
        "user-1",
        {
            "roles": ["admin"],
            "projects": ["foo", "bar"],
        },
    )
    await routing_table.register_model("model-1", provider_model_id="model-1", provider_id="test_provider")
    await routing_table.register_model("model-2", provider_model_id="model-2", provider_id="test_provider")
    await routing_table.register_model("model-3", provider_model_id="model-3", provider_id="test_provider")
    model = await routing_table.get_model("test_provider/model-1")
    assert model.identifier == "test_provider/model-1"
    model = await routing_table.get_model("test_provider/model-2")
    assert model.identifier == "test_provider/model-2"
    model = await routing_table.get_model("test_provider/model-3")
    assert model.identifier == "test_provider/model-3"

    mock_get_authenticated_user.return_value = User(
        "user-2",
        {
            "roles": ["user"],
            "projects": ["foo"],
        },
    )
    model = await routing_table.get_model("test_provider/model-1")
    assert model.identifier == "test_provider/model-1"
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("test_provider/model-2")
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("test_provider/model-3")
    with pytest.raises(AccessDeniedError):
        await routing_table.register_model("model-4", provider_id="test_provider")
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("test_provider/model-1")

    mock_get_authenticated_user.return_value = User(
        "user-3",
        {
            "roles": ["user"],
            "projects": ["bar"],
        },
    )
    model = await routing_table.get_model("test_provider/model-2")
    assert model.identifier == "test_provider/model-2"
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("test_provider/model-1")
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("test_provider/model-3")
    with pytest.raises(AccessDeniedError):
        await routing_table.register_model("model-5", provider_id="test_provider")
    with pytest.raises(AccessDeniedError):
        await routing_table.unregister_model("test_provider/model-2")

    mock_get_authenticated_user.return_value = User(
        "user-1",
        {
            "roles": ["admin"],
            "projects": ["foo", "bar"],
        },
    )
    await routing_table.unregister_model("test_provider/model-3")
    with pytest.raises(ModelNotFoundError):
        await routing_table.get_model("test_provider/model-3")


def test_permit_when():
    config = """
    - permit:
        principal: user-1
        actions: [read]
      when: user in owners namespaces
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert not is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_permit_unless():
    # permit unless both conditions are met
    config = """
    - permit:
        principal: user-1
        actions: [read]
        resource: model::*
      unless:
        - user not in owners namespaces
        - user in owners teams
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"], "teams": ["ml-team"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"], "teams": ["ml-team"]}))
    assert not is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_forbid_when():
    config = """
    - forbid:
        principal: user-1
        actions: [read]
      when:
        user in owners namespaces
    - permit:
        actions: [read]
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_forbid_unless():
    config = """
    - forbid:
        principal: user-1
        actions: [read]
      unless:
        user in owners namespaces
    - permit:
        actions: [read]
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("testuser", {"namespaces": ["foo"]}),
    )
    assert is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-1", {"namespaces": ["bar"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"namespaces": ["foo"]}))


def test_user_has_attribute():
    config = """
    - permit:
        actions: [read]
      when: user with admin in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_user_does_not_have_attribute():
    config = """
    - permit:
        actions: [read]
      unless: user with admin not in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_is_owner():
    config = """
    - permit:
        actions: [read]
      when: user is owner
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("user-2", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_is_not_owner():
    config = """
    - permit:
        actions: [read]
      unless: user is not owner
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))
    model = ModelWithOwner(
        identifier="mymodel",
        provider_id="myprovider",
        model_type=ModelType.llm,
        owner=User("user-2", {"namespaces": ["foo"]}),
    )
    assert not is_action_allowed(policy, "read", model, User("user-1", {"roles": ["basic"]}))
    assert is_action_allowed(policy, "read", model, User("user-2", {"roles": ["admin"]}))
    assert not is_action_allowed(policy, "read", model, User("user-3", {"namespaces": ["foo"]}))
    assert not is_action_allowed(policy, "read", model, User("user-4", None))


def test_invalid_rule_permit_and_forbid_both_specified():
    config = """
    - permit:
        actions: [read]
      forbid:
        actions: [create]
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_rule_neither_permit_or_forbid_specified():
    config = """
    - when: user is owner
      unless: user with admin in roles
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_rule_when_and_unless_both_specified():
    config = """
    - permit:
        actions: [read]
      when: user is owner
      unless: user with admin in roles
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


def test_invalid_condition():
    config = """
    - permit:
        actions: [read]
      when: random words that are not valid
    """
    with pytest.raises(ValidationError):
        TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))


@pytest.mark.parametrize(
    "condition",
    [
        "user is owner",
        "user is not owner",
        "user with dev in teams",
        "user with default not in namespaces",
        "user in owners roles",
        "user not in owners projects",
    ],
)
def test_condition_reprs(condition):
    from ogx.core.access_control.conditions import parse_condition

    assert condition == str(parse_condition(condition))


def test_regex_resource_matching():
    """Test regex pattern matching for resources"""
    from ogx.core.access_control.access_control import matches_resource

    # Test exact match (backward compatibility)
    assert matches_resource("model::llama-3-1b", "model::llama-3-1b")

    # Test wildcard (backward compatibility)
    assert matches_resource("model::*", "model::llama-3-1b")

    # Test regex patterns
    assert matches_resource("regex:model::(llama|mistral)-.*", "model::llama-3-1b")
    assert matches_resource("regex:model::(llama|mistral)-.*", "model::mistral-7b")
    assert matches_resource("regex:model::llama-3\\.\\d+b", "model::llama-3.1b")
    assert matches_resource("regex:model::.*-\\d+b$", "model::llama-3-1b")

    # Test that regex doesn't match when it shouldn't
    assert not matches_resource("regex:model::llama-3$", "model::llama-3-1b")
    assert not matches_resource("regex:model::mistral-.*", "model::llama-3-1b")


def test_access_policy_with_regex():
    """Test access policy with regex resource patterns"""
    config = """
    - permit:
        actions: [read]
        resource: regex:model::(llama|mistral)-.*
      when: user with developer in roles
    - permit:
        actions: [read]
        resource: regex:model::.*-3\\.\\d+-.*
      when: user with user in roles
    """
    policy = TypeAdapter(list[AccessRule]).validate_python(yaml.safe_load(config))

    # Developer can access both llama and mistral models
    llama_model = ModelWithOwner(
        identifier="llama-3-1b",
        provider_id="test",
        model_type=ModelType.llm,
    )
    mistral_model = ModelWithOwner(
        identifier="mistral-7b",
        provider_id="test",
        model_type=ModelType.llm,
    )

    assert is_action_allowed(policy, "read", llama_model, User("dev-user", {"roles": ["developer"]}))
    assert is_action_allowed(policy, "read", mistral_model, User("dev-user", {"roles": ["developer"]}))

    # Regular user can only access llama-3.x models
    llama_3_2_model = ModelWithOwner(
        identifier="llama-3.2-8b",
        provider_id="test",
        model_type=ModelType.llm,
    )
    assert is_action_allowed(policy, "read", llama_3_2_model, User("regular-user", {"roles": ["user"]}))
    assert not is_action_allowed(policy, "read", mistral_model, User("regular-user", {"roles": ["user"]}))


@pytest.fixture
def restricted_user():
    """User with limited access."""
    return User("restricted-user", {"roles": ["user"]})


@pytest.fixture
def admin_user():
    """User with admin access."""
    return User("admin-user", {"roles": ["admin"]})


@pytest.fixture
def rbac_policy():
    """RBAC policy that restricts access to certain models."""
    from ogx.core.access_control.datatypes import Action, Scope

    return [
        # Admins get full access
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["user with admin in roles"],
        ),
        # Regular users only get read access to their own resources
        AccessRule(
            permit=Scope(actions=[Action.READ]),
            when=["user is owner"],
        ),
    ]


class TestInferenceRouterRBACBypass:
    """Test RBAC bypass vulnerability in inference router fallback path."""

    @pytest.fixture
    def mock_routing_table(self):
        """Create a mock routing table for testing."""
        routing_table = AsyncMock()
        routing_table.impls_by_provider_id = {"test-provider": AsyncMock()}
        routing_table.policy = []
        return routing_table

    @patch("ogx.core.routers.inference.get_authenticated_user")
    async def test_registry_path_and_fallback_path_consistent(
        self, mock_get_user, mock_routing_table, restricted_user, admin_user, rbac_policy
    ):
        """Test that registry path and fallback path have consistent RBAC enforcement."""
        mock_routing_table.policy = rbac_policy

        # Create a model owned by admin
        admin_model = ModelWithOwner(
            identifier="admin-model",
            provider_id="test-provider",
            provider_resource_id="admin-resource",
            model_type=ModelType.llm,
            type="model",
            metadata={},
            owner=admin_user,
        )

        # Setup router
        router = InferenceRouter(
            routing_table=mock_routing_table,
            store=None,
        )

        # Test 1: Restricted user tries to access via registry (should fail)
        mock_get_user.return_value = restricted_user
        mock_routing_table.get_object_by_identifier.return_value = None  # RBAC blocks it
        with pytest.raises(ModelNotFoundError):
            await router._get_model_provider("admin-model", "llm")

        # Test 2: Restricted user tries to access via fallback path (should also fail)
        mock_routing_table.get_object_by_identifier.return_value = None
        with pytest.raises(ModelNotFoundError):
            await router._get_model_provider("test-provider/admin-resource", "llm")

        # Test 3: Admin user can access via registry
        mock_get_user.return_value = admin_user
        mock_routing_table.get_object_by_identifier.return_value = admin_model
        provider_mock = AsyncMock()
        mock_routing_table.get_provider_impl.return_value = provider_mock

        provider, resource_id = await router._get_model_provider("admin-model", "llm")
        assert provider == provider_mock
        assert resource_id == "admin-resource"

        # Test 4: Admin user can also access via fallback path
        mock_routing_table.get_object_by_identifier.return_value = None
        provider, resource_id = await router._get_model_provider("test-provider/admin-resource", "llm")
        assert provider == mock_routing_table.impls_by_provider_id["test-provider"]
        assert resource_id == "admin-resource"


class TestModelListingRBACBypass:
    """Test RBAC bypass vulnerability in dynamic model listing via provider_data."""

    @patch("ogx.core.routing_tables.models.instantiate_class_type")
    @patch("ogx.core.routing_tables.models.PROVIDER_DATA_VAR")
    @patch("ogx.core.routing_tables.models.get_authenticated_user")
    @patch("ogx.core.routing_tables.common.get_authenticated_user")
    async def test_dynamic_models_respect_rbac(
        self,
        mock_get_user_common,
        mock_get_user_models,
        mock_provider_data,
        mock_instantiate_class,
        cached_disk_dist_registry,
        rbac_policy,
        admin_user,
        restricted_user,
    ):
        """Test that models fetched via provider_data are filtered by RBAC."""
        from ogx.core.request_headers import NeedsRequestProviderData

        # Create a mock provider that supports provider_data
        mock_provider = Mock(spec=NeedsRequestProviderData)
        mock_provider.__provider_spec__ = MagicMock()
        mock_provider.__provider_spec__.api = Api.inference
        mock_provider.__provider_spec__.provider_data_validator = "dict"

        # Mock the validator to always succeed
        mock_validator = MagicMock(return_value={})
        mock_instantiate_class.return_value = mock_validator

        # Mock list_models to return dynamic models
        # These are fetched via provider_data and don't have owners initially
        dynamic_model1 = Model(
            identifier="dynamic-model-1",
            provider_id="test-provider",
            provider_resource_id="dynamic-model-1",
            model_type=ModelType.llm,
            metadata={},
        )
        dynamic_model2 = Model(
            identifier="dynamic-model-2",
            provider_id="test-provider",
            provider_resource_id="dynamic-model-2",
            model_type=ModelType.llm,
            metadata={},
        )
        mock_provider.list_models = AsyncMock(return_value=[dynamic_model1, dynamic_model2])

        # Setup routing table with policy (no models pre-registered in registry)
        routing_table = ModelsRoutingTable(
            impls_by_provider_id={"test-provider": mock_provider},
            dist_registry=cached_disk_dist_registry,
            policy=rbac_policy,
        )

        # Set up provider_data context (user has credentials for this provider)
        mock_provider_data.get.return_value = {"api_key": "test-key"}

        # Test 1: Admin user can see dynamic models
        # Admin rule allows all actions, so they can see models even without ownership
        mock_get_user_common.return_value = admin_user
        mock_get_user_models.return_value = admin_user

        result = await routing_table.list_models()
        model_ids = [m.identifier for m in result.data]
        assert "test-provider/dynamic-model-1" in model_ids
        assert "test-provider/dynamic-model-2" in model_ids

        # Test 2: Restricted user CANNOT see dynamic models
        # Dynamic models have no owner, and policy requires either admin role OR ownership
        # This demonstrates the fix: before, these would be returned without RBAC checks
        mock_get_user_common.return_value = restricted_user
        mock_get_user_models.return_value = restricted_user

        result = await routing_table.list_models()
        model_ids = [m.identifier for m in result.data]
        # Restricted user should see no models (no ownership, not admin)
        assert len(model_ids) == 0


def test_invalid_regex_pattern_in_access_policy_logs_warning(caplog):
    """Test that invalid regex patterns in access_policy log a warning and don't crash"""
    import logging  # allow-direct-logging

    from ogx.core.access_control.access_control import matches_resource

    # Test various invalid regex patterns
    invalid_patterns = [
        "regex:**",  # Invalid: ** is not valid regex
        "regex:(?P<invalid",  # Invalid: unclosed group
        "regex:[z-a]",  # Invalid: bad character range
    ]

    with caplog.at_level(logging.WARNING):
        for pattern in invalid_patterns:
            # Should not crash, should return False (no match)
            result = matches_resource(pattern, "model::test/model-1")
            assert result is False

    # Check that warnings were logged
    assert "Invalid regex pattern in access_policy" in caplog.text
