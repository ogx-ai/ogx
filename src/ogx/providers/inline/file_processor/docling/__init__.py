# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import DoclingFileProcessorConfig


async def get_provider_impl(config: DoclingFileProcessorConfig, deps: dict[Api, Any]):
    """Get the Docling file processor implementation."""
    from .docling import DoclingFileProcessor

    assert isinstance(config, DoclingFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps.get(Api.files)

    impl = DoclingFileProcessor(config, files_api)
    return impl


__all__ = ["DoclingFileProcessorConfig", "get_provider_impl"]
