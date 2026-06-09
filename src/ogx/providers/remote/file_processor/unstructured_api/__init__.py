# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import UnstructuredApiFileProcessorConfig


async def get_adapter_impl(config: UnstructuredApiFileProcessorConfig, deps: dict[Api, Any]):
    from .unstructured_api import UnstructuredApiFileProcessor

    assert isinstance(config, UnstructuredApiFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps.get(Api.files)
    if files_api is None:
        raise ValueError(
            "Failed to find required dependency: files API is required for unstructured-api file processor"
        )

    impl = UnstructuredApiFileProcessor(config, files_api)
    return impl


__all__ = ["UnstructuredApiFileProcessorConfig", "get_adapter_impl"]
