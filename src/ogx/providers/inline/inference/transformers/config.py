# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class TransformersInferenceConfig(BaseModel):
    """Configuration for the transformers inference provider."""

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {}
