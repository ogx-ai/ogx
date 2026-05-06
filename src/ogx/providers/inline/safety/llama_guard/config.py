# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class LlamaGuardConfig(BaseModel):
    """Configuration for the Llama Guard safety provider with category exclusion settings."""

    excluded_categories: list[str] = []

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "excluded_categories": [],
        }
