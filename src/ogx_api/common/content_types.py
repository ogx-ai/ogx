# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from ogx_api.schema_utils import json_schema_type, register_schema


@json_schema_type
class URL(BaseModel):
    """A URL reference to external content.

    :param uri: The URL string pointing to the resource
    """

    uri: str


class _URLOrData(BaseModel):
    """
    A URL or a base64 encoded string

    :param url: A URL of the image or data URL in the format of data:image/{type};base64,{data}. Note that URL could have length limits.
    :param data: base64 encoded image data as string
    """

    url: URL | None = None
    # data is a base64 encoded string, hint with contentEncoding=base64
    data: str | None = Field(default=None, json_schema_extra={"contentEncoding": "base64"})

    @model_validator(mode="before")
    @classmethod
    def validator(cls, values: Any) -> dict[str, Any]:
        if isinstance(values, dict):
            return values
        return {"url": values}


@json_schema_type
class ImageContentItem(BaseModel):
    """A image content item

    :param type: Discriminator type of the content item. Always "image"
    :param image: Image as a base64 encoded string or an URL
    """

    type: Literal["image"] = "image"
    image: _URLOrData


@json_schema_type
class TextContentItem(BaseModel):
    """A text content item

    :param type: Discriminator type of the content item. Always "text"
    :param text: Text content
    """

    type: Literal["text"] = "text"
    text: str


# other modalities can be added here
InterleavedContentItem = Annotated[
    ImageContentItem | TextContentItem,
    Field(discriminator="type"),
]
register_schema(InterleavedContentItem, name="InterleavedContentItem")

# accept a single "str" as a special case since it is common
InterleavedContent = str | InterleavedContentItem | list[InterleavedContentItem]
register_schema(InterleavedContent, name="InterleavedContent")


@json_schema_type
class TextDelta(BaseModel):
    """A text content delta for streaming responses.

    :param type: Discriminator type of the delta. Always "text"
    :param text: The incremental text content
    """

    type: Literal["text"] = "text"
    text: str


@json_schema_type
class ImageDelta(BaseModel):
    """An image content delta for streaming responses.

    :param type: Discriminator type of the delta. Always "image"
    :param image: The incremental image data as bytes
    """

    type: Literal["image"] = "image"
    image: bytes
