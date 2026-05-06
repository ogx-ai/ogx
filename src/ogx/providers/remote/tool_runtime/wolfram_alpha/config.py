# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class WolframAlphaToolConfig(BaseModel):
    """Configuration for WolframAlpha Tool Runtime"""

    api_key: str | None = None

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${env.WOLFRAM_ALPHA_API_KEY:=}",
        }
