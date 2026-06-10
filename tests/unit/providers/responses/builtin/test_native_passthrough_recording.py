# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Record / replay coverage for the native vLLM /v1/responses passthrough stream."""

from unittest.mock import patch


def _load_native_sse_fixture() -> list[str]:
    """Real vLLM /v1/responses SSE stream (reasoning + file_search function_call +
    completed-with-usage), captured live and committed as a fixture so the native
    passthrough can be exercised deterministically without a running vLLM."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "vllm_native_file_search_stream.json"
    return json.loads(fixture.read_text())


def _make_vllm_adapter():
    """Duck-typed stand-in carrying just what VLLMInferenceAdapter._stream_response
    touches on ``self`` (avoids constructing the real pydantic-model adapter)."""
    from types import SimpleNamespace

    from pydantic import TypeAdapter

    from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
    from ogx_api import OpenAIResponseObjectStream

    return SimpleNamespace(
        _build_httpx_client_kwargs=lambda: {},
        _response_stream_adapter=TypeAdapter(OpenAIResponseObjectStream),
        _normalize_response_object=VLLMInferenceAdapter._normalize_response_object,
    )


def _mock_backend_stream(sse_lines: list[str]):
    """Sync callable matching httpx.AsyncClient.stream: returns an async context
    manager whose response replays the given SSE lines."""
    import httpx as _httpx

    class _CM:
        async def __aenter__(self):
            class _Resp:
                status_code = 200
                headers = _httpx.Headers({"content-type": "text/event-stream"})

                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    for line in sse_lines:
                        yield line

            return _Resp()

        async def __aexit__(self, *args):
            return False

    def _stream(self, method, url, **kwargs):
        return _CM()

    return _stream


async def _drive_native_stream(adapter):
    from ogx.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter

    endpoint = "http://vllm.test:8000/v1/responses"
    payload = {"model": "openai/gpt-oss-20b", "input": "hi", "stream": True}
    events = []
    iterator = await VLLMInferenceAdapter._stream_response(adapter, endpoint, {}, payload)
    async for event in iterator:
        events.append(event)
    return events


async def test_native_responses_stream_recording_roundtrip(tmp_path):
    """The api_recorder must intercept the native /v1/responses httpx stream:
    record it live, then replay it without touching the backend, and the parsed
    completed event must still carry usage (exercises the adapter's response_id /
    store / text normalization on real vLLM event shapes)."""
    import httpx

    from ogx.testing.api_recorder import APIRecordingMode, ResponseStorage, api_recording

    sse_lines = _load_native_sse_fixture()
    storage_dir = tmp_path / "native_responses_roundtrip"

    def _has_usage(events):
        for event in events:
            resp = getattr(event, "response", None)
            if resp is not None and getattr(resp, "usage", None) is not None:
                return resp.usage
        return None

    # Record: backend mock is the "original" the recorder wraps.
    with patch("httpx.AsyncClient.stream", new=_mock_backend_stream(sse_lines)):
        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage_dir)):
            recorded_events = await _drive_native_stream(_make_vllm_adapter())

    assert ResponseStorage(storage_dir)._get_test_dir().exists()
    usage = _has_usage(recorded_events)
    assert usage is not None, "completed event with usage was dropped during recording"
    assert usage.output_tokens_details.reasoning_tokens > 0

    # Replay: backend must NOT be called; recorded SSE drives the adapter.
    backend = httpx.AsyncClient.stream
    with patch("httpx.AsyncClient.stream", wraps=backend) as backend_mock:
        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage_dir)):
            replayed_events = await _drive_native_stream(_make_vllm_adapter())
        backend_mock.assert_not_called()

    replayed_usage = _has_usage(replayed_events)
    assert replayed_usage is not None, "completed event with usage missing on replay"
    assert replayed_usage.total_tokens == usage.total_tokens
    assert any(
        type(e).__name__ == "OpenAIResponseObjectStreamResponseOutputItemAdded"
        and getattr(getattr(e, "item", None), "type", None) == "function_call"
        for e in replayed_events
    ), "file_search function_call event missing on replay"
