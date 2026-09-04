# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api

from .config import OpenSearchVectorIOConfig


async def get_adapter_impl(
    config: OpenSearchVectorIOConfig, deps: dict[Api, Any], policy: list[AccessRule] | None = None
):
    from .opensearch import OpenSearchVectorIOAdapter

    impl = OpenSearchVectorIOAdapter(
        config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors), policy=policy or []
    )
    await impl.initialize()
    return impl
