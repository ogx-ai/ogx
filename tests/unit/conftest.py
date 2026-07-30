# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging  # allow-direct-logging
import os
import warnings

import pytest


def pytest_sessionstart(session) -> None:
    if "OGX_LOGGING" not in os.environ:
        os.environ["OGX_LOGGING"] = "all=WARNING"

    # Silence common deprecation spam during unit tests.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(autouse=True)
def suppress_httpx_logs(caplog):
    """Suppress httpx INFO logs for all unit tests"""
    caplog.set_level(logging.WARNING, logger="httpx")
