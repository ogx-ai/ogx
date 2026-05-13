# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import httpx

from ogx.providers.remote.inference.watsonx.config import WatsonXConfig
from ogx.providers.remote.inference.watsonx.watsonx import WatsonXInferenceAdapter


class _FailingIamClient:
    def __init__(self, calls: list[int]) -> None:
        self._calls = calls

    async def __aenter__(self) -> "_FailingIamClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, *args, **kwargs):
        self._calls.append(1)
        await asyncio.sleep(0.05)
        raise httpx.ConnectError("boom")


async def test_refresh_iam_token_deduplicates_concurrent_failures(monkeypatch):
    adapter = WatsonXInferenceAdapter(
        config=WatsonXConfig(base_url="https://us-south.ml.cloud.ibm.com"),
    )
    calls: list[int] = []

    monkeypatch.setattr(
        "ogx.providers.remote.inference.watsonx.watsonx.httpx.AsyncClient",
        lambda: _FailingIamClient(calls),
    )

    results = await asyncio.gather(*(adapter._refresh_iam_token("watsonx-api-key") for _ in range(3)))

    assert results == ["watsonx-api-key", "watsonx-api-key", "watsonx-api-key"]
    assert len(calls) == 1
