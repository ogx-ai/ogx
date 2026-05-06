# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class MessagesConfig(BaseModel):
    """Configuration for the built-in Anthropic Messages API adapter."""

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "") -> dict[str, Any]:
        return {}
