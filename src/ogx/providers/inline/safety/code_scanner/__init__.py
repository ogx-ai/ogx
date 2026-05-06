# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import CodeScannerConfig


async def get_provider_impl(config: CodeScannerConfig, deps: dict[str, Any]):
    from .code_scanner import BuiltinCodeScannerSafetyImpl

    impl = BuiltinCodeScannerSafetyImpl(config, deps)
    await impl.initialize()
    return impl
