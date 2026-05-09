# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import PromptGuardConfig


async def get_provider_impl(config: PromptGuardConfig, deps: dict[str, Any]):
    from .prompt_guard import PromptGuardSafetyImpl

    impl = PromptGuardSafetyImpl(config, deps)
    await impl.initialize()
    return impl
