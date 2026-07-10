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
    def __init__(self, calls: list[int], client_kwargs: dict | None = None) -> None:
        self._calls = calls
        self.client_kwargs = client_kwargs or {}

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
        lambda **kwargs: _FailingIamClient(calls),
    )

    results = await asyncio.gather(*(adapter._refresh_iam_token("watsonx-api-key") for _ in range(3)))

    assert results == ["watsonx-api-key", "watsonx-api-key", "watsonx-api-key"]
    assert len(calls) == 1


async def test_iam_token_exchange_applies_network_tls_config(monkeypatch):
    """IAM token exchange builds its httpx client with the provider's network config (issue #6251)."""
    from ogx.providers.utils.inference.model_registry import NetworkConfig, TLSConfig

    adapter = WatsonXInferenceAdapter(
        config=WatsonXConfig(
            base_url="https://us-south.ml.cloud.ibm.com",
            network=NetworkConfig(tls=TLSConfig(verify=False)),
        ),
    )
    captured: dict = {}

    def _make_client(**kwargs):
        captured.update(kwargs)
        return _FailingIamClient([], kwargs)

    monkeypatch.setattr(
        "ogx.providers.remote.inference.watsonx.watsonx.httpx.AsyncClient",
        _make_client,
    )

    await adapter._exchange_iam_token("watsonx-api-key")

    # The network config must reach the client (verify=False -> a "verify" kwarg is present),
    # rather than the client being constructed with no TLS arguments.
    assert "verify" in captured
