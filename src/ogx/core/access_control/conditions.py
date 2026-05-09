# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol


class User(Protocol):
    """Protocol for user identity with principal and attribute information."""

    principal: str
    attributes: dict[str, list[str]] | None


class ProtectedResource(Protocol):
    """Protocol for resources subject to access control."""

    type: str
    identifier: str
    owner: User | None


class Condition(Protocol):
    """Protocol for access control conditions that evaluate resource-user relationships."""

    def matches(self, resource: ProtectedResource, user: User) -> bool: ...


class UserInOwnersList:
    """Condition that checks if the user has any matching attribute values in the resource owner's attribute list."""

    def __init__(self, name: str):
        self.name = name

    def owners_values(self, resource: ProtectedResource) -> list[str] | None:
        if (
            hasattr(resource, "owner")
            and resource.owner
            and resource.owner.attributes
            and self.name in resource.owner.attributes
        ):
            return resource.owner.attributes[self.name]
        else:
            return None

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        defined = self.owners_values(resource)
        if not defined:
            return False
        if not user.attributes or self.name not in user.attributes or not user.attributes[self.name]:
            return False
        user_values = user.attributes[self.name]
        for value in defined:
            if value in user_values:
                return True
        return False

    def __repr__(self) -> str:
        return f"user in owners {self.name}"


class UserNotInOwnersList(UserInOwnersList):
    """Condition that checks if the user does NOT have matching attribute values in the resource owner's attribute list."""

    def __init__(self, name: str):
        super().__init__(name)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not super().matches(resource, user)

    def __repr__(self) -> str:
        return f"user not in owners {self.name}"


class UserWithValueInList:
    """Condition that checks if the user has a specific value in a named attribute list."""

    def __init__(self, name: str, value: str):
        self.name = name
        self.value = value

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        if user.attributes and self.name in user.attributes:
            return self.value in user.attributes[self.name]
        print(f"User does not have {self.value} in {self.name}")
        return False

    def __repr__(self) -> str:
        return f"user with {self.value} in {self.name}"


class UserWithValueNotInList(UserWithValueInList):
    """Condition that checks if the user does NOT have a specific value in a named attribute list."""

    def __init__(self, name: str, value: str):
        super().__init__(name, value)

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not super().matches(resource, user)

    def __repr__(self) -> str:
        return f"user with {self.value} not in {self.name}"


class UserIsOwner:
    """Condition that checks if the user is the owner of the resource."""

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return resource.owner.principal == user.principal if resource.owner else False

    def __repr__(self) -> str:
        return "user is owner"


class UserIsNotOwner:
    """Condition that checks if the user is NOT the owner of the resource."""

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not resource.owner or resource.owner.principal != user.principal

    def __repr__(self) -> str:
        return "user is not owner"


class ResourceIsUnowned:
    """Condition that checks if the resource has no owner."""

    def matches(self, resource: ProtectedResource, user: User) -> bool:
        return not resource.owner

    def __repr__(self) -> str:
        return "resource is unowned"


def parse_condition(condition: str) -> Condition:
    """Parse a condition string into a Condition object.

    Args:
        condition: A natural language condition string (e.g., 'user is owner', 'user in owners roles').

    Returns:
        A Condition instance matching the parsed expression.

    Raises:
        ValueError: If the condition string is not recognized.
    """
    words = condition.split()
    match words:
        case ["user", "is", "owner"]:
            return UserIsOwner()
        case ["user", "is", "not", "owner"]:
            return UserIsNotOwner()
        case ["user", "with", value, "in", name]:
            return UserWithValueInList(name, value)
        case ["user", "with", value, "not", "in", name]:
            return UserWithValueNotInList(name, value)
        case ["user", "in", "owners", name]:
            return UserInOwnersList(name)
        case ["user", "not", "in", "owners", name]:
            return UserNotInOwnersList(name)
        case ["resource", "is", "unowned"]:
            return ResourceIsUnowned()
        case _:
            raise ValueError(f"Invalid condition: {condition}")


def parse_conditions(conditions: list[str]) -> list[Condition]:
    """Parse a list of condition strings into Condition objects.

    Args:
        conditions: List of natural language condition strings.

    Returns:
        List of corresponding Condition instances.
    """
    return [parse_condition(c) for c in conditions]
