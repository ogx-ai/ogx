# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import DatabricksImplConfig


async def get_adapter_impl(config: DatabricksImplConfig, _deps):
    from .databricks import DatabricksInferenceAdapter

    assert isinstance(config, DatabricksImplConfig), f"Unexpected config type: {type(config)}"
    impl = DatabricksInferenceAdapter(config=config)
    await impl.initialize()
    return impl
