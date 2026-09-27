# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""KV-backed persistence for Gemini thought_signature ↔ OpenAI tool call id."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta

from ogx.log import get_logger
from ogx.providers.remote.inference.vertexai.converters import _normalize_thought_signature
from ogx_api.internal.kvstore import KVStore

logger = get_logger(__name__, category="inference")

_KEY_PREFIX = "thought_sig:"
# Gemini only validates signatures within the current tool-call turn. 24h covers
# multi-step loops, restarts, slow tools, and brief human-in-the-loop pauses
# without unbounded KV growth.
_DEFAULT_TTL = timedelta(hours=24)


class ThoughtSignatureStore:
    """Persist Gemini thought_signature values keyed by OpenAI tool call id."""

    def __init__(self, kv: KVStore, ttl: timedelta = _DEFAULT_TTL) -> None:
        self._kv = kv
        self._ttl = ttl

    def _key(self, call_id: str) -> str:
        return f"{_KEY_PREFIX}{call_id}"

    def _expiration(self) -> datetime:
        return datetime.now(tz=UTC) + self._ttl

    async def put(self, call_id: str, signature: str | None) -> None:
        normalized = _normalize_thought_signature(signature)
        if not normalized or not call_id:
            return
        await self._kv.set(self._key(call_id), normalized, expiration=self._expiration())
        logger.debug(
            "Persisted thought_signature",
            call_id=call_id,
            ttl_seconds=int(self._ttl.total_seconds()),
        )

    async def put_many(self, signatures: Mapping[str, str]) -> None:
        for call_id, signature in signatures.items():
            await self.put(call_id, signature)

    async def get(self, call_id: str) -> str | None:
        if not call_id:
            return None
        value = await self._kv.get(self._key(call_id))
        if value is None:
            logger.debug("thought_signature cache miss", call_id=call_id)
            return None
        logger.debug("thought_signature cache hit", call_id=call_id)
        return value

    async def get_many(self, call_ids: Iterable[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for call_id in call_ids:
            if not call_id or call_id in result:
                continue
            value = await self.get(call_id)
            if value is not None:
                result[call_id] = value
        return result
