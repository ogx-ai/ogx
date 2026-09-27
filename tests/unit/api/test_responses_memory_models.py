# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx_api.responses.models import CreateResponseRequest, MemoryToolConfig


def test_create_response_accepts_memory_disabled():
    request = CreateResponseRequest(input="hi", model="test", memory={"enabled": False})

    assert request.memory == MemoryToolConfig(enabled=False)


def test_memory_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        MemoryToolConfig(enabled=True, unsafe=True)


def test_memory_config_bounds_max_num_results():
    with pytest.raises(ValidationError):
        MemoryToolConfig(max_num_results=51)
