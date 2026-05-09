# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test provider-data validators for secret handling.
Categories: configuration validation.
Specific tests: ensure secret-bearing provider data fields use SecretStr.
"""

from typing import get_args, get_origin

import pytest
from pydantic import BaseModel, SecretStr

from ogx.core.distribution import get_provider_registry, providable_apis
from ogx.core.utils.dynamic import instantiate_class_type


def _is_secretstr(annotation) -> bool:
    if annotation is SecretStr:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return SecretStr in get_args(annotation)


def _looks_secret(name: str) -> bool:
    lowered = name.lower()
    return "key" in lowered or "token" in lowered or "credential" in lowered


class TestProviderDataSecretStr:
    @pytest.mark.parametrize("api", providable_apis())
    def test_provider_data_fields_are_secretstr(self, api):
        registry = get_provider_registry()
        providers = registry.get(api, {})

        failures: list[str] = []

        for provider_type, spec in providers.items():
            validator_path = spec.provider_data_validator
            if not validator_path:
                continue

            validator_cls = instantiate_class_type(validator_path)
            if not issubclass(validator_cls, BaseModel):
                failures.append(f"{provider_type}: provider_data_validator is not a pydantic BaseModel")
                continue

            for field_name, field_info in validator_cls.model_fields.items():
                if not _looks_secret(field_name):
                    continue
                if not _is_secretstr(field_info.annotation):
                    failures.append(
                        f"{provider_type}: field '{field_name}' should be SecretStr (or Optional[SecretStr])"
                    )

        if failures:
            pytest.fail("\n".join(failures))
