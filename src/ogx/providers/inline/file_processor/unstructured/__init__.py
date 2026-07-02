# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import UnstructuredFileProcessorConfig


async def get_provider_impl(config: UnstructuredFileProcessorConfig, deps: dict[Api, Any]):
    """Get the Unstructured file processor implementation."""
    from .unstructured import UnstructuredFileProcessor

    assert isinstance(config, UnstructuredFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps[Api.files]

    impl = UnstructuredFileProcessor(config, files_api)
    return impl


__all__ = ["UnstructuredFileProcessorConfig", "get_provider_impl"]
