# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import DoclingServeFileProcessorConfig


async def get_adapter_impl(config: DoclingServeFileProcessorConfig, deps: dict[Api, Any]):
    from .docling_serve import DoclingServeFileProcessor

    assert isinstance(config, DoclingServeFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps.get(Api.files)
    if files_api is None:
        raise ValueError("Failed to find required dependency: files API is required for docling-serve file processor")

    impl = DoclingServeFileProcessor(config, files_api)
    return impl


__all__ = ["DoclingServeFileProcessorConfig", "get_adapter_impl"]
