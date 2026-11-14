# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class PlaceholderWebSearchToolConfig(BaseModel):
    """Configuration for the placeholder web search tool runtime.

    The placeholder has no options: it performs no search and answers
    invocations with a notice that web search is not configured.
    """

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}
