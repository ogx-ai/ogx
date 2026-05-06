# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import Api

from .config import InteractionsConfig


async def get_provider_impl(
    config: InteractionsConfig,
    deps: dict[Api, Any],
):
    from .impl import BuiltinInteractionsImpl

    impl = BuiltinInteractionsImpl(config, deps[Api.inference])
    await impl.initialize()
    return impl
