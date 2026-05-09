# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, model_validator

from .conditions import parse_conditions


class Action(StrEnum):
    """CRUD actions for access control evaluation."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"


class Scope(BaseModel):
    """Defines the scope of an access rule including optional principal, actions, and resource constraints."""

    principal: str | None = None
    actions: Action | list[Action]
    resource: str | None = None


class RouteScope(BaseModel):
    """Scope for route-level access control.

    Defines which API routes can be accessed. The paths field
    accepts single paths, lists of paths, wildcards, or regex patterns:
    - Exact: "/v1/chat/completions"
    - List: ["/v1/files*", "/v1/models*"]
    - Prefix wildcard: "/v1/files*" matches "/v1/files" and all paths starting with "/v1/files"
    - Full wildcard: "*"
    - Regex pattern: "regex:/v1/(chat|inference)/.*" matches paths using regular expressions
    """

    paths: str | list[str]


def _mutually_exclusive(obj: Any, a: str, b: str) -> None:
    if getattr(obj, a) and getattr(obj, b):
        raise ValueError(f"{a} and {b} are mutually exclusive")


def _require_one_of(obj: Any, a: str, b: str) -> None:
    if not getattr(obj, a) and not getattr(obj, b):
        raise ValueError(f"on of {a} or {b} is required")


class AccessRule(BaseModel):
    """Access rule based loosely on cedar policy language

    A rule defines a list of action either to permit or to forbid. It may specify a
    principal or a resource that must match for the rule to take effect. The resource
    to match should be specified in the form of a type qualified identifier, e.g.
    model::my-model or vector_store::some-db, a wildcard for all resources of a type,
    e.g. model::*, or a regex pattern with the "regex:" prefix, e.g.
    regex:model::(llama|mistral)-3\\.\\d+-\\d+b. If the principal or resource are not
    specified, they will match all requests.

    A rule may also specify a condition, either a 'when' or an 'unless', with additional
    constraints as to where the rule applies. The constraints supported at present are:

    - 'user with <attr-value> in <attr-name>'
    - 'user with <attr-value> not in <attr-name>'
    - 'user is owner'
    - 'user is not owner'
    - 'user in owners <attr-name>'
    - 'user not in owners <attr-name>'

    Rules are tested in order to find a match. If a match is found, the request is
    permitted or forbidden depending on the type of rule. If no match is found, the
    request is denied. If no rules are specified, a rule that allows any action as
    long as the resource attributes match the user attributes is added
    (i.e. the previous behaviour is the default).

    Some examples in yaml:

    - permit:
        principal: user-1
        actions: [create, read, delete]
        resource: model::*
      description: user-1 has full access to all models
    - permit:
        principal: user-2
        actions: [read]
        resource: model::model-1
      description: user-2 has read access to model-1 only
    - permit:
        actions: [read]
      when: user in owner teams
      description: any user has read access to any resource created by a member of their team
    - forbid:
        actions: [create, read, delete]
        resource: vector_store::*
      unless: user with admin in roles
      description: only user with admin role can use vector_store resources

    """

    permit: Scope | None = None
    forbid: Scope | None = None
    when: str | list[str] | None = None
    unless: str | list[str] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_rule_format(self) -> Self:
        _require_one_of(self, "permit", "forbid")
        _mutually_exclusive(self, "permit", "forbid")
        _mutually_exclusive(self, "when", "unless")
        if isinstance(self.when, list):
            parse_conditions(self.when)
        elif self.when:
            parse_conditions([self.when])
        if isinstance(self.unless, list):
            parse_conditions(self.unless)
        elif self.unless:
            parse_conditions([self.unless])
        return self


class RouteAccessRule(BaseModel):
    """Route-level access rule for controlling API route access.

    This rule defines which API routes users can access based on their
    attributes (roles, teams, etc). Rules are evaluated before resource-level
    access control.

    A rule defines either permit or forbid access to specific routes. The routes
    are specified using the 'paths' field which can be:
    - A single exact path: "/v1/chat/completions"
    - A list of paths: ["/v1/files*", "/v1/models*"]
    - A wildcard prefix: "/v1/files*" matches /v1/files and all paths starting with /v1/files
    - Full wildcard: "*" matches all routes
    - Regex pattern: "regex:/v1/(chat|inference)/.*" matches paths using regular expressions

    Path normalization: Trailing slashes are automatically removed (e.g., /v1/files/ becomes /v1/files).

    A rule may also specify a condition using 'when' or 'unless', with the same
    constraints as resource-level rules:
    - 'user with <attr-value> in <attr-name>'
    - 'user with <attr-value> not in <attr-name>'

    If no route_policy is configured, all routes are allowed.
    If route_policy is configured, rules are tested in order to find a match.

    Examples in yaml:

    - permit:
        paths: "/v1/chat/completions"
      when: user with developer in roles
      description: developers can access chat completions

    - permit:
        paths: ["/v1/files*", "/v1/models*"]
      when: user with user in roles
      description: users can access files and models routes

    - permit:
        paths: "*"
      when: user with admin in roles
      description: admins have access to all routes
    """

    permit: RouteScope | None = None
    forbid: RouteScope | None = None
    when: str | list[str] | None = None
    unless: str | list[str] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_rule_format(self) -> Self:
        _require_one_of(self, "permit", "forbid")
        _mutually_exclusive(self, "permit", "forbid")
        _mutually_exclusive(self, "when", "unless")
        if isinstance(self.when, list):
            parse_conditions(self.when)
        elif self.when:
            parse_conditions([self.when])
        if isinstance(self.unless, list):
            parse_conditions(self.unless)
        elif self.unless:
            parse_conditions([self.unless])
        return self
