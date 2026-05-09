# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    ApiFilter,
    HealthInfo,
    ListRoutesResponse,
    VersionInfo,
)


@runtime_checkable
class Inspect(Protocol):
    """APIs for inspecting the OGX service, including health status, available API routes with methods and implementing providers."""

    async def list_routes(self, api_filter: ApiFilter | None = None) -> ListRoutesResponse: ...

    async def health(self) -> HealthInfo: ...

    async def version(self) -> VersionInfo: ...
