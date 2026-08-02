# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from ogx_api import Conversation, SamplingStrategy
from ogx_api.schema_utils import (
    clear_dynamic_schema_types,
    flatten_nullable_remove_default,
    get_registered_schema_info,
    iter_dynamic_schema_types,
    iter_json_schema_types,
    iter_registered_schema_types,
    register_dynamic_schema_type,
    remove_default_from_schema,
    remove_null_from_anyof,
)


def test_json_schema_registry_contains_known_model() -> None:
    assert Conversation in iter_json_schema_types()


def test_registered_schema_registry_contains_sampling_strategy() -> None:
    registered_names = {info.name for info in iter_registered_schema_types()}
    assert "SamplingStrategy" in registered_names

    schema_info = get_registered_schema_info(SamplingStrategy)
    assert schema_info is not None
    assert schema_info.name == "SamplingStrategy"


class TestRemoveNullFromAnyof:
    def test_flattens_single_type_with_null(self) -> None:
        schema: dict = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
        remove_null_from_anyof(schema)
        assert schema == {"type": "boolean"}

    def test_preserves_non_null_properties(self) -> None:
        schema: dict = {"anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]}
        remove_null_from_anyof(schema)
        assert schema == {"type": "string", "enum": ["a", "b"]}

    def test_no_anyof_is_noop(self) -> None:
        schema: dict = {"type": "string"}
        remove_null_from_anyof(schema)
        assert schema == {"type": "string"}

    def test_no_null_variant_is_noop(self) -> None:
        schema: dict = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        remove_null_from_anyof(schema)
        assert schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    def test_add_nullable_flag(self) -> None:
        schema: dict = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
        remove_null_from_anyof(schema, add_nullable=True)
        assert schema == {"type": "boolean", "nullable": True}

    def test_empty_anyof(self) -> None:
        schema: dict = {"anyOf": []}
        remove_null_from_anyof(schema)
        assert schema == {"anyOf": []}


class TestRemoveDefaultFromSchema:
    def test_removes_existing_default(self) -> None:
        schema: dict = {"type": "boolean", "default": True}
        remove_default_from_schema(schema)
        assert schema == {"type": "boolean"}

    def test_no_default_is_noop(self) -> None:
        schema: dict = {"type": "string"}
        remove_default_from_schema(schema)
        assert schema == {"type": "string"}

    def test_removes_none_default(self) -> None:
        schema: dict = {"type": "string", "default": None}
        remove_default_from_schema(schema)
        assert schema == {"type": "string"}


class TestFlattenNullableRemoveDefault:
    def test_removes_null_and_default(self) -> None:
        schema: dict = {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": True}
        flatten_nullable_remove_default(schema)
        assert schema == {"type": "boolean"}

    def test_only_null_removal_when_no_default(self) -> None:
        schema: dict = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        flatten_nullable_remove_default(schema)
        assert schema == {"type": "string"}

    def test_only_default_removal_when_no_anyof(self) -> None:
        schema: dict = {"type": "integer", "default": 0}
        flatten_nullable_remove_default(schema)
        assert schema == {"type": "integer"}

    def test_empty_schema(self) -> None:
        schema: dict = {}
        flatten_nullable_remove_default(schema)
        assert schema == {}


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
