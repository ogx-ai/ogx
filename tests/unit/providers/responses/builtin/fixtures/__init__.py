# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import yaml

from ogx_api.inference import (
    OpenAIChatCompletion,
)

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def load_chat_completion_fixture(filename: str) -> OpenAIChatCompletion:
    fixture_path = os.path.join(FIXTURES_DIR, filename)

    with open(fixture_path) as f:
        data = yaml.safe_load(f)
    return OpenAIChatCompletion(**data)
