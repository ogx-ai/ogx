# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import TypeSafeConfig


async def get_adapter_impl(config: TypeSafeConfig, _deps: Any = None):
    from .typesafe import TypeSafeInferenceAdapter

    if not isinstance(config, TypeSafeConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")
    adapter = TypeSafeInferenceAdapter(config=config)
    await adapter.initialize()
    return adapter


__all__ = ["get_adapter_impl", "TypeSafeConfig"]
