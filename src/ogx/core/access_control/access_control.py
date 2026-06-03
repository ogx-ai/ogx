# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from typing import Any

from ogx.core.datatypes import User
from ogx.log import get_logger

from .conditions import (
    Condition,
    ProtectedResource,
    parse_conditions,
)
from .datatypes import (
    AccessRule,
    Action,
    Scope,
)

logger = get_logger(name=__name__, category="core::auth")

ALLOWED_ATTRIBUTE_KEYS = frozenset({"roles", "teams", "projects", "namespaces"})


def matches_resource(resource_scope: str, actual_resource: str) -> bool:
    """Check if a resource scope pattern matches an actual resource identifier.

    Args:
        resource_scope: A scope pattern (exact match, wildcard with ::*, or regex: prefix).
        actual_resource: The qualified resource identifier to match against.

    Returns:
        True if the resource matches the scope pattern.
    """
    if resource_scope == actual_resource:
        return True
    if resource_scope.startswith("regex:"):
        pattern = resource_scope[6:]
        try:
            return bool(re.match(pattern, actual_resource))
        except re.error as e:
            logger.warning(
                "Invalid regex pattern in access_policy, treating as non-match", pattern=pattern, error=str(e)
            )
            return False
    return resource_scope.endswith("::*") and actual_resource.startswith(resource_scope[:-1])


def matches_scope(
    scope: Scope,
    action: Action,
    resource: str,
    user: str | None,
) -> bool:
    """Check if a scope matches the given action, resource, and user principal.

    Args:
        scope: The access control scope to check against.
        action: The action being performed.
        resource: The qualified resource identifier.
        user: The user principal, or None.

    Returns:
        True if the scope matches all provided criteria.
    """
    if scope.resource and not matches_resource(scope.resource, resource):
        return False
    if scope.principal and scope.principal != user:
        return False
    return action in scope.actions


def as_list(obj: Any) -> list[Any]:
    """Wrap a value in a list if it is not already a list.

    Args:
        obj: A value or list of values.

    Returns:
        The value wrapped in a list, or the original list.
    """
    if isinstance(obj, list):
        return obj
    return [obj]


def matches_conditions(
    conditions: list[Condition],
    resource: ProtectedResource,
    user: User,
) -> bool:
    """Check if all conditions in a list are satisfied for the given resource and user.

    Args:
        conditions: List of conditions that must all match.
        resource: The protected resource being accessed.
        user: The user attempting access.

    Returns:
        True if all conditions match, False if any condition fails.
    """
    for condition in conditions:
        # must match all conditions
        if not condition.matches(resource, user):
            return False
    return True


def default_policy() -> list[AccessRule]:
    """Return the default access control policy for backwards compatibility.

    Returns:
        A list of AccessRules that permit all actions when user attributes match resource owner attributes.
    """
    # for backwards compatibility, if no rules are provided, assume
    # full access subject to previous attribute matching rules
    return [
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["user in owners " + name],
        )
        for name in ALLOWED_ATTRIBUTE_KEYS
    ] + [
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["user is owner"],
        ),
        AccessRule(
            permit=Scope(actions=list(Action)),
            when=["resource is unowned"],
        ),
    ]


def is_action_allowed(
    policy: list[AccessRule],
    action: Action,
    resource: ProtectedResource,
    user: User | None,
) -> bool:
    """Evaluate access control policy to determine if an action is allowed.

    Args:
        policy: List of access rules to evaluate in order.
        action: The action being attempted.
        resource: The protected resource being accessed.
        user: The authenticated user, or None if authentication is disabled.

    Returns:
        True if the action is allowed, False otherwise.
    """
    qualified_resource_id = f"{resource.type}::{resource.identifier}"
    decision = False
    reason = ""
    index = -1

    # If user is not set, assume authentication is not enabled
    if not user:
        decision = True
        reason = "no auth"
    else:
        if not len(policy):
            policy = default_policy()

        for index, rule in enumerate(policy):  # noqa: B007
            if rule.forbid and matches_scope(rule.forbid, action, qualified_resource_id, user.principal):
                if rule.when:
                    if matches_conditions(parse_conditions(as_list(rule.when)), resource, user):
                        decision = False
                        reason = rule.description or ""
                        break
                elif rule.unless:
                    if not matches_conditions(parse_conditions(as_list(rule.unless)), resource, user):
                        decision = False
                        reason = rule.description or ""
                        break
                else:
                    decision = False
                    reason = rule.description or ""
                    break
            elif rule.permit and matches_scope(rule.permit, action, qualified_resource_id, user.principal):
                if rule.when:
                    if matches_conditions(parse_conditions(as_list(rule.when)), resource, user):
                        decision = True
                        reason = rule.description or ""
                        break
                elif rule.unless:
                    if not matches_conditions(parse_conditions(as_list(rule.unless)), resource, user):
                        decision = True
                        reason = rule.description or ""
                        break
                else:
                    decision = True
                    reason = rule.description or ""
                    break
        else:
            reason = "no matching rule"
            index = -1

    # print apprived or denied
    decision_str = "APPROVED" if decision else "DENIED"
    user_str = user.principal if user else "none"
    logger.debug(
        "AUTHZ",
        decision_str=decision_str,
        user_str=user_str,
        qualified_resource_id=qualified_resource_id,
        action=action,
        index=index,
        reason=reason,
    )
    return decision


class AccessDeniedError(RuntimeError):
    """Raised when a user is denied access to perform an action on a resource."""

    def __init__(self, action: str | None = None, resource: ProtectedResource | None = None, user: User | None = None):
        self.action = action
        self.resource = resource
        self.user = user

        message = _build_access_denied_message(action, resource, user)
        super().__init__(message)


def _build_access_denied_message(action: str | None, resource: ProtectedResource | None, user: User | None) -> str:
    """Build detailed error message for access denied scenarios."""
    if action and resource and user:
        resource_info = f"{resource.type}::{resource.identifier}"
        user_info = f"'{user.principal}'"
        if user.attributes:
            attrs = ", ".join([f"{k}={v}" for k, v in user.attributes.items()])
            user_info += f" (attributes: {attrs})"

        message = f"User {user_info} cannot perform action '{action}' on resource '{resource_info}'"
    else:
        message = "Insufficient permissions"

    return message
