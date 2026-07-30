# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable

IdFactory = Callable[[], str]
IdOverride = Callable[[str, IdFactory], str]

_id_override: IdOverride | None = None


def generate_object_id(kind: str, factory: IdFactory) -> str:
    """Generate an identifier for the given kind using the provided factory.

    Allows tests to override ID generation deterministically by installing an
    override callback via :func:`set_id_override`.
    """

    override = _id_override
    if override is not None:
        return override(kind, factory)
    return factory()


def set_id_override(override: IdOverride) -> IdOverride | None:
    """Install an override used to generate deterministic identifiers."""

    global _id_override

    previous = _id_override
    _id_override = override
    return previous


def reset_id_override(previous: IdOverride | None) -> None:
    """Restore the previous override returned by :func:`set_id_override`."""

    global _id_override
    _id_override = previous
