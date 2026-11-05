# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type

DEFAULT_BASE_URL = "https://api.typesafe.ai"


class TypeSafeProviderDataValidator(BaseModel):
    """Validates provider-specific request data for the TypeSafe inference provider."""

    typesafe_api_key: SecretStr | None = Field(
        default=None,
        description="API key for TypeSafe, overriding the configured key for this request.",
    )


@json_schema_type
class TypeSafeConfig(RemoteInferenceProviderConfig):
    """Configuration for the TypeSafe SystemOne (judgment/decision) inference provider.

    Backs ``POST /v1/systemone`` (judgment calls) and the ``ModelType.systemone`` model
    registry through TypeSafe's decision-model API. Inherits ``allowed_models``,
    ``refresh_models``, ``api_key`` (auth credential) and ``network`` from the remote
    inference config base.
    """

    base_url: HttpUrl = Field(
        default=HttpUrl(DEFAULT_BASE_URL),
        description="Base URL for the TypeSafe API.",
    )

    @classmethod
    def sample_run_config(
        cls,
        api_key: str = "${env.TYPESAFE_API_KEY:=}",
        base_url: str = "${env.TYPESAFE_BASE_URL:=" + DEFAULT_BASE_URL + "}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
            "api_key": api_key,
        }
