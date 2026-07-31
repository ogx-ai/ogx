# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Query
from pydantic import BaseModel

from ogx.core.utils.type_inspection import is_body_param, is_unwrapped_body_param


class SampleModel(BaseModel):
    name: str


def test_is_body_param_with_body_annotation() -> None:
    assert is_body_param(Annotated[SampleModel, Body(...)]) is True


def test_is_body_param_with_query_annotation() -> None:
    assert is_body_param(Annotated[str, Query()]) is False


def test_is_body_param_with_plain_type() -> None:
    assert is_body_param(str) is False
    assert is_body_param(SampleModel) is False


def test_is_unwrapped_body_param_with_default_embed() -> None:
    # Body(...) has embed=None by default, should be treated as unwrapped
    assert is_unwrapped_body_param(Annotated[SampleModel, Body(...)]) is True


def test_is_unwrapped_body_param_with_embed_false() -> None:
    assert is_unwrapped_body_param(Annotated[SampleModel, Body(embed=False)]) is True


def test_is_unwrapped_body_param_with_embed_true() -> None:
    assert is_unwrapped_body_param(Annotated[SampleModel, Body(embed=True)]) is False
