# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MetaConfig


async def get_adapter_impl(config: MetaConfig, _deps):
    # import dynamically so the import is used only when it is needed
    from .meta import MetaInferenceAdapter

    adapter = MetaInferenceAdapter(config=config)
    return adapter
