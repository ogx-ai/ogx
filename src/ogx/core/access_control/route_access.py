# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared route-level access control evaluation.

This is the single source of truth for "is this user allowed to access this
route" -- used by RouteAuthorizationMiddleware for live HTTP/WebSocket requests,
and by any component (e.g. the batches provider) that executes a request against
an API route outside of that middleware's reach, such as in-process background
processing. Both call sites must agree, or a route forbidden at the HTTP layer
can be reached indirectly.
"""

import re

from ogx.core.access_control.conditions import User as ProtocolUser
from ogx.core.access_control.conditions import parse_conditions
from ogx.core.access_control.datatypes import RouteAccessRule
from ogx.core.datatypes import User
from ogx.log import get_logger

logger = get_logger(name=__name__, category="core::access_control")


class _RouteContext:
    """Placeholder resource for route-level condition evaluation.

    Route rules don't operate on actual resources, so we use this context object
    to satisfy the condition.matches() interface. Route conditions typically check
    user attributes (e.g., "user with admin in roles") and don't require resource properties.
    """

    def __init__(self) -> None:
        self.type = "route"
        self.identifier = "route"
        self.owner: ProtocolUser | None = None


def _route_matches(request_route: str, rule_patterns: str | list[str]) -> bool:
    """Check if request route matches any of the rule patterns.

    Supports:
    - Exact match: "/v1/chat/completions"
    - Prefix wildcard: "/v1/files*" matches "/v1/files", "/v1/files/upload", "/v1/files/list", etc.
    - Full wildcard: "*" matches all routes
    - Regex pattern: "regex:/v1/(chat|inference)/.*" matches routes using regular expressions
    """
    patterns = [rule_patterns] if isinstance(rule_patterns, str) else rule_patterns

    for pattern in patterns:
        if pattern == "*":
            # Full wildcard matches everything
            return True
        elif pattern.startswith("regex:"):
            # Regex pattern: extract pattern after "regex:" prefix
            regex_pattern = pattern[6:]
            try:
                if re.match(regex_pattern, request_route):
                    return True
            except re.error as e:
                logger.warning(
                    "Invalid regex pattern in route_policy, skipping this pattern.",
                    regex_pattern=regex_pattern,
                    error=str(e),
                )
        elif pattern.endswith("*"):
            # Prefix wildcard: check if request route starts with the prefix
            prefix = pattern[:-1]  # Remove "*"
            if request_route.startswith(prefix):
                return True
        elif pattern == request_route:
            # Exact match
            return True

    return False


def _evaluate_conditions(rule: RouteAccessRule, user: User | None) -> bool:
    """Evaluate when/unless conditions for the rule.

    Reuses the existing condition parsing from access_control.conditions.

    Args:
        rule: The rule whose conditions to evaluate
        user: The authenticated user, or None if no authentication is configured

    Returns:
        True if conditions are met (or no conditions exist), False otherwise
    """
    # If rule has conditions but no user is available, conditions cannot be met
    if (rule.when or rule.unless) and not user:
        return False

    if rule.when:
        # At this point, if rule.when exists and we got past the check above,
        # user is guaranteed to be non-None
        assert user is not None
        conditions_list = rule.when if isinstance(rule.when, list) else [rule.when]
        conditions = parse_conditions(conditions_list)
        # For 'when', all conditions must match (AND logic)
        # Note: Since we're checking route access, we don't have a resource,
        # so we create a context object to satisfy the interface
        route_context = _RouteContext()
        for condition in conditions:
            if not condition.matches(route_context, user):
                return False
        return True

    if rule.unless:
        # At this point, if rule.unless exists and we got past the check above,
        # user is guaranteed to be non-None
        assert user is not None
        conditions_list = rule.unless if isinstance(rule.unless, list) else [rule.unless]
        conditions = parse_conditions(conditions_list)
        # For 'unless', no conditions should match (NOT logic)
        route_context = _RouteContext()
        for condition in conditions:
            if condition.matches(route_context, user):
                return False
        return True

    # No conditions specified - rule applies regardless of user
    return True


def _rule_matches(rule: RouteAccessRule, route: str, user: User | None) -> bool:
    """Check if a rule matches the given route and user.

    Args:
        rule: The rule to evaluate
        route: The route being accessed
        user: The authenticated user, or None if no authentication is configured
    """
    # Get the scope (permit or forbid)
    scope = rule.permit if rule.permit else rule.forbid
    if not scope:
        return False

    # Check if route matches
    if not _route_matches(route, scope.paths):
        return False

    # Evaluate conditions
    return _evaluate_conditions(rule, user)


def is_route_allowed(route: str, user: User | None, route_policy: list[RouteAccessRule]) -> bool:
    """Check if the user is allowed to access the given route.

    Rules are evaluated in order. First matching rule determines access.
    If no rule matches, access is denied. If ``route_policy`` is empty, every
    route is allowed (backward compatible -- route-level authorization is opt-in).

    Args:
        route: The route being accessed, e.g. "/v1/chat/completions"
        user: The authenticated user, or None if no authentication is configured
        route_policy: The configured route access rules (``server.auth.route_policy``)
    """
    if not route_policy:
        return True

    user_str = user.principal if user else "anonymous"

    for index, rule in enumerate(route_policy):
        if _rule_matches(rule, route, user):
            # Check if this is a permit or forbid rule
            if rule.permit:
                decision = "APPROVED"
                reason = rule.description or ""
                logger.debug(
                    "ROUTE_AUTHZ",
                    decision=decision,
                    user_str=user_str,
                    route=route,
                    index=index,
                    reason=reason,
                )
                return True
            else:  # forbid
                decision = "DENIED"
                reason = rule.description or ""
                logger.debug(
                    "ROUTE_AUTHZ",
                    decision=decision,
                    user_str=user_str,
                    route=route,
                    index=index,
                    reason=reason,
                )
                return False

    # No matching rule found - deny by default
    logger.debug("ROUTE_AUTHZ", decision="DENIED", user=user_str, route=route, rule_index=-1, reason="no matching rule")
    return False
