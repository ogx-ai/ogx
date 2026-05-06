# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import PyPDFFileProcessorConfig


async def get_provider_impl(config: PyPDFFileProcessorConfig, deps: dict[Api, Any]):
    """Get the PyPDF file processor implementation."""
    from .adapter import PyPDFFileProcessorAdapter

    assert isinstance(config, PyPDFFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps[Api.files]

    impl = PyPDFFileProcessorAdapter(config, files_api)
    return impl


__all__ = ["PyPDFFileProcessorConfig", "get_provider_impl"]
