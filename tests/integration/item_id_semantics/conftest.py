# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Fixtures and xfail policy for the item-id semantics suite.

Two targets:

- local OGX server (default, like the rest of the integration suite) — parity check
- live OpenAI API — reference check:

    pytest tests/integration/item_id_semantics --live-openai
    OGX_LIVE_OPENAI=1 pytest tests/integration/item_id_semantics

Divergence policy: tests marked ``@pytest.mark.divergence("Dn", reason)`` pin
the OpenAI reference behavior. On the local target the marker is converted in
``pytest_collection_modifyitems`` into a NON-STRICT xfail, so the tests run in
full and any expected failure — the actual divergence, or a missing replay
recording — converts to XP instead of a failure. Once the parity fixes land,
the tests pass and report XPASS automatically; the tests themselves never
need editing. On live OpenAI no xfail is applied: the tests must pass.
"""

import os

import pytest
from openai import OpenAI

# Imported so the fixture is visible in this directory (local-mode target).
from tests.integration.fixtures.common import openai_client  # noqa: F401


def is_live_openai(config) -> bool:
    """Whether the live-OpenAI target is selected (CLI flag or env var)."""
    if config.getoption("--live-openai", default=False):
        return True
    return os.environ.get("OGX_LIVE_OPENAI") == "1"


def pytest_collection_modifyitems(config, items):
    if is_live_openai(config):
        return
    for item in items:
        for mark in item.iter_markers(name="divergence"):
            code, reason = mark.args
            item.add_marker(
                pytest.mark.xfail(
                    strict=False,
                    reason=f"OGX divergence {code}: {reason} (non-strict; XPASS once fixed)",
                )
            )


@pytest.fixture(scope="session")
def id_sem_client(request):
    """OpenAI-compatible client pointed at live OpenAI or the local server."""
    if is_live_openai(request.config):
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
