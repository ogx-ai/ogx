# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ogx_api.schema_utils import json_schema_type, register_schema


@json_schema_type
class StringType(BaseModel):
    """Parameter type for string values.

    :param type: Discriminator type. Always "string"
    """

    type: Literal["string"] = "string"


@json_schema_type
class NumberType(BaseModel):
    """Parameter type for numeric values.

    :param type: Discriminator type. Always "number"
    """

    type: Literal["number"] = "number"


@json_schema_type
class BooleanType(BaseModel):
    """Parameter type for boolean values.

    :param type: Discriminator type. Always "boolean"
    """

    type: Literal["boolean"] = "boolean"


@json_schema_type
class ArrayType(BaseModel):
    """Parameter type for array values.

    :param type: Discriminator type. Always "array"
    """

    type: Literal["array"] = "array"


@json_schema_type
class ObjectType(BaseModel):
    """Parameter type for object values.

    :param type: Discriminator type. Always "object"
    """

    type: Literal["object"] = "object"


@json_schema_type
class JsonType(BaseModel):
    """Parameter type for JSON values.

    :param type: Discriminator type. Always "json"
    """

    type: Literal["json"] = "json"


@json_schema_type
class UnionType(BaseModel):
    """Parameter type for union values.

    :param type: Discriminator type. Always "union"
    """

    type: Literal["union"] = "union"


@json_schema_type
class ChatCompletionInputType(BaseModel):
    """Parameter type for chat completion input.

    :param type: Discriminator type. Always "chat_completion_input"
    """

    # expects List[Message] for messages
    type: Literal["chat_completion_input"] = "chat_completion_input"


@json_schema_type
class CompletionInputType(BaseModel):
    """Parameter type for completion input.

    :param type: Discriminator type. Always "completion_input"
    """

    # expects InterleavedTextMedia for content
    type: Literal["completion_input"] = "completion_input"


@json_schema_type
class DialogType(BaseModel):
    """Parameter type for dialog data with semantic output labels.

    :param type: Discriminator type. Always "dialog"
    """

    # expects List[Message] for messages
    # this type semantically contains the output label whereas ChatCompletionInputType does not
    type: Literal["dialog"] = "dialog"


ParamType = Annotated[
    StringType
    | NumberType
    | BooleanType
    | ArrayType
    | ObjectType
    | JsonType
    | UnionType
    | ChatCompletionInputType
    | CompletionInputType,
    Field(discriminator="type"),
]
register_schema(ParamType, name="ParamType")

"""
# TODO: recursive definition of ParamType in these containers
# will cause infinite recursion in OpenAPI generation script
# since we are going with ChatCompletionInputType and CompletionInputType
# we don't need to worry about ArrayType/ObjectType/UnionType for now
ArrayType.model_rebuild()
ObjectType.model_rebuild()
UnionType.model_rebuild()


class CustomType(BaseModel):
pylint: disable=syntax-error
    type: Literal["custom"] = "custom"
    validator_class: str
"""
