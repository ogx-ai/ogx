# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import pathlib


class TestCase:
    _apis = [
        "inference/chat_completion",
        "inference/completion",
    ]
    _jsonblob = {}

    def __init__(self, name):
        # loading all test cases
        if self._jsonblob == {}:
            for api in self._apis:
                with open(pathlib.Path(__file__).parent / f"{api}.json") as f:
                    coloned = api.replace("/", ":")
                    try:
                        loaded = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"There is a syntax error in {api}.json: {e}") from e
                    TestCase._jsonblob.update({f"{coloned}:{k}": v for k, v in loaded.items()})

        # loading this test case
        tc = self._jsonblob.get(name)
        if tc is None:
            raise ValueError(f"Test case {name} not found")

        # these are the only fields we need
        self.data = tc.get("data")

    def __getitem__(self, key):
        return self.data[key]
