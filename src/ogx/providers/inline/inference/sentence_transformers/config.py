# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field


class SentenceTransformersInferenceConfig(BaseModel):
    """Configuration for the sentence-transformers inference provider."""

    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust and execute remote code from model repositories. "
        "Set to True for models that require custom code (e.g., nomic-ai/nomic-embed-text-v1.5). "
        "Defaults to False for security.",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {"trust_remote_code": False}
