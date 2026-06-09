# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import AutoFileProcessorConfig


async def get_provider_impl(config: AutoFileProcessorConfig, deps: dict[Api, Any]):
    """Get the auto file processor implementation."""
    from .auto import AutoFileProcessor

    assert isinstance(config, AutoFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps[Api.files]

    impl = AutoFileProcessor(config, files_api)
    return impl


__all__ = ["AutoFileProcessorConfig", "get_provider_impl"]
