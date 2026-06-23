# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return the list of available container_runtime provider specifications.

    The container_runtime API backs the public Containers API: providers
    implement the backend lifecycle (Docker/Podman, Kubernetes) that the
    Containers service delegates to. No backends ship in-tree yet, so the
    registry is currently empty.
    """
    return []
