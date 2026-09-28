# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import tempfile
from pathlib import Path

import httpx
import pytest

from ogx.testing.api_recorder import (
    APIRecordingMode,
    _is_tei_model_lookup_url,
    api_recording,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test recordings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestTeiModelLookupInterception:
    """The TEI adapter discovers its single served model via a raw ``httpx.AsyncClient`` GET to
    the server's native ``/info`` endpoint (``server_signature.get_text_embeddings_inference_model_id``)
    -- TEI has no ``/v1/models`` endpoint, so the OpenAI-SDK patch never sees this request. The
    recorder must capture it as a model lookup so replay CI can discover the model with no
    backend running.
    """

    TEI_INFO_URL = "http://localhost:8080/info"
    TEI_INFO_BODY = {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "served_model_name": "nomic-ai/nomic-embed-text-v1.5",
    }

    @staticmethod
    def _backend(body: dict, calls: list | None = None) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if calls is not None:
                calls.append(request)
            return httpx.Response(200, json=body)

        return httpx.MockTransport(handler)

    @staticmethod
    def _no_backend() -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no backend is listening", request=request)

        return httpx.MockTransport(handler)

    @staticmethod
    def _recordings(storage: Path) -> list[Path]:
        return sorted(storage.rglob("*.json"))

    @pytest.mark.parametrize(
        "url,intercepted",
        [
            ("http://localhost:8080/info", True),
            ("http://127.0.0.1:8080/info", True),
            ("http://localhost:8080/v1/info", False),
            ("http://localhost:8080/health", False),
            ("http://localhost:8080/props", False),
            ("http://localhost:8080/v1/models", False),
        ],
    )
    def test_tei_model_lookup_predicate(self, url, intercepted):
        """Only the server-root /info model lookup is intercepted; other GET targets (including
        llama.cpp's /props signature check) pass straight through."""
        assert _is_tei_model_lookup_url(url) is intercepted

    async def test_tei_info_get_is_recorded_as_a_model_list(self, temp_storage_dir):
        """Record mode stores the /info response under the model-list naming convention
        (models-*.json) so it is found by the model-list lookup paths."""
        storage = temp_storage_dir / "tei_record"
        calls: list[httpx.Request] = []

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.TEI_INFO_BODY, calls)) as client:
                response = await client.get(self.TEI_INFO_URL)

        assert response.json() == self.TEI_INFO_BODY

        recordings = self._recordings(storage)
        assert len(recordings) == 1
        assert recordings[0].name.startswith("models-")
        recorded = json.loads(recordings[0].read_text())
        assert recorded["request"]["method"] == "GET"
        assert recorded["request"]["url"] == self.TEI_INFO_URL
        assert recorded["request"]["endpoint"] == "/info"
        assert recorded["response"]["body"] == self.TEI_INFO_BODY

    async def test_tei_info_get_replays_with_no_backend_running(self, temp_storage_dir):
        """Replay mode must serve the recording without reaching the network -- this is the
        whole point: replay CI has no Text-Embeddings-Inference server."""
        storage = temp_storage_dir / "tei_replay"
        calls: list[httpx.Request] = []

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.TEI_INFO_BODY, calls)) as client:
                await client.get(self.TEI_INFO_URL)
        assert len(calls) == 1

        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._no_backend()) as client:
                replayed = await client.get(self.TEI_INFO_URL)

        # Still one call: replay answered from disk, the dead transport was never touched.
        assert len(calls) == 1
        assert isinstance(replayed, httpx.Response)
        assert replayed.status_code == 200
        assert replayed.json() == self.TEI_INFO_BODY

    async def test_signature_checks_replay_from_the_same_recording(self, temp_storage_dir):
        """The adapter's model lookup and its /info signature checks (initialize, health) issue
        the identical GET /info request, so the single recording must satisfy both through the
        real server_signature code paths, with no backend running."""
        from ogx.providers.utils.inference.server_signature import (
            check_text_embeddings_inference_server,
            get_text_embeddings_inference_model_id,
        )
        from ogx_api import HealthStatus

        storage = temp_storage_dir / "tei_signature_replay"

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.TEI_INFO_BODY)) as client:
                await client.get(self.TEI_INFO_URL)

        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage)):
            model_id = await get_text_embeddings_inference_model_id("http://localhost:8080/v1")
            health = await check_text_embeddings_inference_server("http://localhost:8080/v1")

        assert model_id == "nomic-ai/nomic-embed-text-v1.5"
        assert health["status"] == HealthStatus.OK

    async def test_different_served_models_produce_distinct_recordings(self, temp_storage_dir):
        """The filename digest covers the served model, so re-recording with a different TEI
        model adds a file instead of overwriting the previous model set."""
        storage = temp_storage_dir / "tei_models"

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend({"model_id": "model-a"})) as client:
                await client.get(self.TEI_INFO_URL)

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend({"model_id": "model-b"})) as client:
                await client.get(self.TEI_INFO_URL)

        recordings = self._recordings(storage)
        assert len(recordings) == 2

    async def test_record_if_missing_skips_write_for_unchanged_model(self, temp_storage_dir):
        """Re-recording the same served model in record-if-missing mode must not rewrite the
        existing file (same write-skip behaviour as the /v1/models union)."""
        storage = temp_storage_dir / "tei_rim"

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.TEI_INFO_BODY)) as client:
                await client.get(self.TEI_INFO_URL)

        recordings = self._recordings(storage)
        mtime = (storage / "recordings" / recordings[0].name).stat().st_mtime_ns

        with api_recording(mode=APIRecordingMode.RECORD_IF_MISSING, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.TEI_INFO_BODY)) as client:
                response = await client.get(self.TEI_INFO_URL)

        assert response.json() == self.TEI_INFO_BODY
        assert len(self._recordings(storage)) == 1
        assert (storage / "recordings" / self._recordings(storage)[0].name).stat().st_mtime_ns == mtime

    async def test_missing_tei_info_recording_reports_how_to_record(self, temp_storage_dir):
        """A missing recording must fail with the recorder's own instructions, not a
        ``ConnectError`` from falling through to the network."""
        storage = temp_storage_dir / "tei_missing"

        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._no_backend()) as client:
                with pytest.raises(RuntimeError, match="Recording not found for TEI model lookup"):
                    await client.get(self.TEI_INFO_URL)

    async def test_unrelated_httpx_get_is_not_intercepted(self, temp_storage_dir):
        """The predicate stays narrow: llama.cpp's /props signature check and every other GET
        target must pass straight through to the transport."""
        storage = temp_storage_dir / "unrelated_get"
        calls: list[httpx.Request] = []

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend({"total_slots": 1}, calls)) as client:
                await client.get("http://localhost:8080/props")

        assert len(calls) == 1
        assert self._recordings(storage) == []
