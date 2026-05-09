# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Resource-level ABAC evaluation for multitenant isolation.

Tests OGX's attribute-based access control at the resource level
(vector stores, models, etc.) — independent of chunk-level metadata filtering.

Metrics produced:
  - **True positive rate**: correctly permitted access to owned/authorized resources.
  - **True negative rate**: correctly denied access to unauthorized resources.
  - **False positive rate**: incorrectly permitted access (security violation, must be 0).
  - **False negative rate**: incorrectly denied access (usability issue).

Run::

    uv run pytest tests/evals/multitenant/test_resource_access_control.py -v
"""

import pytest

from ogx.core.access_control.access_control import (
    default_policy,
    is_action_allowed,
)
from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.datatypes import User

from .conftest import (
    TENANT_A,
    TENANT_ADMIN,
    TENANT_B,
    UNAUTHORIZED_USER,
)


class _MockResource:
    """Minimal ProtectedResource for testing ABAC policy evaluation."""

    def __init__(self, resource_type: str, identifier: str, owner: User):
        self.type = resource_type
        self.identifier = identifier
        self.owner = owner


# --- Resource fixtures ---

TENANT_A_VECTOR_STORE = _MockResource(
    resource_type="vector_db",
    identifier="acme-financial-docs",
    owner=TENANT_A,
)

TENANT_B_VECTOR_STORE = _MockResource(
    resource_type="vector_db",
    identifier="beta-healthcare-docs",
    owner=TENANT_B,
)

TENANT_A_MODEL = _MockResource(
    resource_type="model",
    identifier="acme-finetuned-llama",
    owner=TENANT_A,
)

PUBLIC_MODEL = _MockResource(
    resource_type="model",
    identifier="llama-3-base",
    owner=None,
)


# ---------------------------------------------------------------------------
# 1. Default policy evaluation
# ---------------------------------------------------------------------------


class TestDefaultPolicyEvaluation:
    """Test the default ABAC policy (attribute matching on owners)."""

    def test_owner_can_read_own_resource(self):
        """Resource owner can always read their own resource."""
        policy = default_policy()
        assert is_action_allowed(policy, Action.READ, TENANT_A_VECTOR_STORE, TENANT_A)

    def test_owner_can_delete_own_resource(self):
        policy = default_policy()
        assert is_action_allowed(policy, Action.DELETE, TENANT_A_VECTOR_STORE, TENANT_A)

    def test_cross_tenant_read_denied(self):
        """Tenant B cannot read Tenant A's vector store under default policy."""
        policy = default_policy()
        assert not is_action_allowed(policy, Action.READ, TENANT_A_VECTOR_STORE, TENANT_B)

    def test_cross_tenant_delete_denied(self):
        policy = default_policy()
        assert not is_action_allowed(policy, Action.DELETE, TENANT_A_VECTOR_STORE, TENANT_B)

    def test_unauthorized_user_denied(self):
        """A user with no matching attributes is denied access."""
        policy = default_policy()
        assert not is_action_allowed(policy, Action.READ, TENANT_A_VECTOR_STORE, UNAUTHORIZED_USER)

    def test_public_resource_accessible(self):
        """Resources with no owner attributes are accessible to all authenticated users."""
        policy = default_policy()
        # Public model has empty attributes, so the default policy
        # (which checks owner attribute intersection) permits access
        assert is_action_allowed(policy, Action.READ, PUBLIC_MODEL, TENANT_A)
        assert is_action_allowed(policy, Action.READ, PUBLIC_MODEL, TENANT_B)

    def test_no_auth_permits_all(self):
        """When authentication is disabled (user=None), all access is permitted."""
        policy = default_policy()
        assert is_action_allowed(policy, Action.READ, TENANT_A_VECTOR_STORE, None)


# ---------------------------------------------------------------------------
# 2. Strict namespace isolation policy
# ---------------------------------------------------------------------------


class TestNamespaceIsolationPolicy:
    """Test a strict policy that isolates tenants by namespace."""

    @pytest.fixture
    def namespace_policy(self):
        return [
            AccessRule(
                permit=Scope(actions=list(Action)),
                when="user in owners namespaces",
                description="permit access only when user shares a namespace with the resource owner",
            ),
        ]

    def test_same_namespace_permitted(self, namespace_policy):
        assert is_action_allowed(namespace_policy, Action.READ, TENANT_A_VECTOR_STORE, TENANT_A)

    def test_different_namespace_denied(self, namespace_policy):
        assert not is_action_allowed(namespace_policy, Action.READ, TENANT_A_VECTOR_STORE, TENANT_B)

    def test_admin_with_both_namespaces(self, namespace_policy):
        """Admin user with both namespaces can access both tenants' resources."""
        assert is_action_allowed(namespace_policy, Action.READ, TENANT_A_VECTOR_STORE, TENANT_ADMIN)
        assert is_action_allowed(namespace_policy, Action.READ, TENANT_B_VECTOR_STORE, TENANT_ADMIN)

    def test_unauthorized_user_no_namespace(self, namespace_policy):
        assert not is_action_allowed(namespace_policy, Action.READ, TENANT_A_VECTOR_STORE, UNAUTHORIZED_USER)


# ---------------------------------------------------------------------------
# 3. Role-based restrictions
# ---------------------------------------------------------------------------


class TestRoleBasedRestrictions:
    """Test policies that restrict actions based on roles."""

    @pytest.fixture
    def admin_only_delete_policy(self):
        return [
            AccessRule(
                forbid=Scope(actions=[Action.DELETE], resource="vector_db::*"),
                unless="user with admin in roles",
                description="only admins can delete vector stores",
            ),
            AccessRule(
                permit=Scope(actions=list(Action)),
                when="user with admin in roles",
                description="admins can perform all actions",
            ),
            AccessRule(
                permit=Scope(actions=[Action.READ, Action.CREATE, Action.UPDATE]),
                when="user in owners namespaces",
                description="read/create/update permitted within namespace",
            ),
        ]

    def test_non_admin_cannot_delete(self, admin_only_delete_policy):
        assert not is_action_allowed(
            admin_only_delete_policy,
            Action.DELETE,
            TENANT_A_VECTOR_STORE,
            TENANT_A,
        )

    def test_admin_can_delete(self, admin_only_delete_policy):
        assert is_action_allowed(
            admin_only_delete_policy,
            Action.DELETE,
            TENANT_A_VECTOR_STORE,
            TENANT_ADMIN,
        )

    def test_non_admin_can_read_own(self, admin_only_delete_policy):
        assert is_action_allowed(
            admin_only_delete_policy,
            Action.READ,
            TENANT_A_VECTOR_STORE,
            TENANT_A,
        )


# ---------------------------------------------------------------------------
# 4. Comprehensive ABAC correctness metrics
# ---------------------------------------------------------------------------


class TestABACCorrectnessMetrics:
    """Compute true positive, true negative, false positive, false negative rates
    across a matrix of (user, resource, action) combinations."""

    def _build_test_matrix(self):
        """Build all (user, resource, action, expected_result) tuples."""
        resources = [
            TENANT_A_VECTOR_STORE,
            TENANT_B_VECTOR_STORE,
            TENANT_A_MODEL,
            PUBLIC_MODEL,
        ]
        users = [TENANT_A, TENANT_B, TENANT_ADMIN, UNAUTHORIZED_USER]
        actions = [Action.READ, Action.CREATE, Action.DELETE]

        # Expected results under default policy:
        # - Owner or attribute-matched user: permitted
        # - Cross-tenant: denied
        # - Public (no owner attributes): permitted for all
        # - Unauthorized: denied for owned resources
        test_cases = []
        for user in users:
            for resource in resources:
                for action in actions:
                    # Determine expected result
                    if not resource.owner:
                        # Public resource: accessible to all
                        expected = True
                    elif user == TENANT_ADMIN:
                        # Admin has all namespaces/teams
                        expected = True
                    elif user.principal == resource.owner.principal:
                        # Owner
                        expected = True
                    elif user == UNAUTHORIZED_USER:
                        expected = False
                    else:
                        # Check attribute overlap
                        user_ns = set(user.attributes.get("namespaces", []))
                        owner_ns = set(resource.owner.attributes.get("namespaces", []))
                        user_teams = set(user.attributes.get("teams", []))
                        owner_teams = set(resource.owner.attributes.get("teams", []))
                        expected = bool(user_ns & owner_ns) and bool(user_teams & owner_teams)

                    test_cases.append((user, resource, action, expected))

        return test_cases

    def test_abac_correctness_metrics(self):
        """Compute and report ABAC correctness metrics."""
        policy = default_policy()
        test_cases = self._build_test_matrix()

        tp, tn, fp, fn = 0, 0, 0, 0

        for user, resource, action, expected in test_cases:
            actual = is_action_allowed(policy, action, resource, user)
            if expected and actual:
                tp += 1
            elif not expected and not actual:
                tn += 1
            elif not expected and actual:
                fp += 1
            elif expected and not actual:
                fn += 1

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        # Critical assertion: zero false positives (no security violations)
        assert fp == 0, (
            f"SECURITY VIOLATION: {fp} false positives detected. Unauthorized access was incorrectly permitted."
        )
        assert fn == 0, f"ABAC denied {fn} legitimate access requests"
        assert accuracy >= 0.95, f"ABAC accuracy {accuracy:.4f} is below threshold"
