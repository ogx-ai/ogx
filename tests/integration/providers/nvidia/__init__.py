# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

# Reusable skip decorator for NVIDIA tests in GitHub Actions
# Adding this in conftest.py as a module level skip statement causes pytest to error
# out in certain cases.
skip_in_github_actions = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true", reason="Skipping NVIDIA tests in GitHub Actions environment"
)
