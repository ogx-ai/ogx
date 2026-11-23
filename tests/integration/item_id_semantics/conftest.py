# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Fixtures for the item-id semantics suite.

Two targets:

- local OGX server (default, like the rest of the integration suite) — parity check
- live OpenAI API — reference check:

    pytest tests/integration/item_id_semantics --live-openai
    OGX_LIVE_OPENAI=1 pytest tests/integration/item_id_semantics
"""

import os

import pytest
from openai import OpenAI

# Imported so the fixture is visible in this directory (local-mode target).
from tests.integration.fixtures.common import openai_client  # noqa: F401


def _is_live_openai(request) -> bool:
    if request.config.getoption("--live-openai", default=False):
        return True
    return os.environ.get("OGX_LIVE_OPENAI") == "1"


@pytest.fixture(scope="session")
def id_sem_client(request):
    """OpenAI-compatible client pointed at live OpenAI or the local server."""
    if _is_live_openai(request):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY is not set; required for live OpenAI runs")
        return OpenAI(api_key=api_key, max_retries=0, timeout=120)
    return request.getfixturevalue("openai_client")


@pytest.fixture(scope="session")
def conversation_registry(request, id_sem_client):
    """Collects conversation ids created by this suite; deletes them (best effort)."""
    ids: list[str] = []
    yield ids
    for conv_id in ids:
        try:
            id_sem_client.conversations.delete(conv_id)
        except Exception:
            pass
