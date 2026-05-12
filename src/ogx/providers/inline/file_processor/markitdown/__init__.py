# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import MarkItDownFileProcessorConfig


async def get_provider_impl(config: MarkItDownFileProcessorConfig, deps: dict[Api, Any]):
    """Get the MarkItDown file processor implementation."""
    from .markitdown_processor import MarkItDownFileProcessor

    assert isinstance(config, MarkItDownFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps[Api.files]

    impl = MarkItDownFileProcessor(config, files_api)
    return impl


__all__ = ["MarkItDownFileProcessorConfig", "get_provider_impl"]
