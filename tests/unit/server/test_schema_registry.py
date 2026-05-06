# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from ogx_api import Conversation, SamplingStrategy
from ogx_api.schema_utils import (
    clear_dynamic_schema_types,
    get_registered_schema_info,
    iter_dynamic_schema_types,
    iter_json_schema_types,
    iter_registered_schema_types,
    register_dynamic_schema_type,
)


def test_json_schema_registry_contains_known_model() -> None:
    assert Conversation in iter_json_schema_types()


def test_registered_schema_registry_contains_sampling_strategy() -> None:
    registered_names = {info.name for info in iter_registered_schema_types()}
    assert "SamplingStrategy" in registered_names

    schema_info = get_registered_schema_info(SamplingStrategy)
    assert schema_info is not None
    assert schema_info.name == "SamplingStrategy"


def test_dynamic_schema_registration_round_trip() -> None:
    existing_models = tuple(iter_dynamic_schema_types())
    clear_dynamic_schema_types()
    try:

        class TemporaryModel(BaseModel):
            foo: str

        register_dynamic_schema_type(TemporaryModel)
        assert TemporaryModel in iter_dynamic_schema_types()

        clear_dynamic_schema_types()
        assert TemporaryModel not in iter_dynamic_schema_types()
    finally:
        for model in existing_models:
            register_dynamic_schema_type(model)
