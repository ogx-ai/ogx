# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI, NotFoundError

from ogx.core.testing_context import reset_test_context, set_test_context
from ogx.testing.api_recorder import (
    APIRecordingMode,
    ResponseStorage,
    _is_ogx_test_server_model_list_url,
    _should_intercept_httpx,
    api_recording,
    normalize_inference_request,
)

# Import the real Pydantic response types instead of using Mocks
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test recordings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def real_openai_chat_response():
    """Real OpenAI chat completion response using proper Pydantic objects."""
    return OpenAIChatCompletion(
        id="chatcmpl-test123",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant", content="Hello! I'm doing well, thank you for asking."
                ),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="llama3.2:3b",
    )


@pytest.fixture
def real_embeddings_response():
    """Real OpenAI embeddings response using proper Pydantic objects."""
    return OpenAIEmbeddingsResponse(
        object="list",
        data=[
            OpenAIEmbeddingData(object="embedding", embedding=[0.1, 0.2, 0.3], index=0),
            OpenAIEmbeddingData(object="embedding", embedding=[0.4, 0.5, 0.6], index=1),
        ],
        model="nomic-embed-text",
        usage=OpenAIEmbeddingUsage(prompt_tokens=6, total_tokens=6),
    )


class TestInferenceRecording:
    """Test the inference recording system."""

    def test_request_normalization(self):
        """Test that request normalization produces consistent hashes."""
        # Test basic normalization
        hash1 = normalize_inference_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        # Same request should produce same hash
        hash2 = normalize_inference_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = normalize_inference_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {
                "model": "llama3.2:3b",
                "messages": [{"role": "user", "content": "Different message"}],
                "temperature": 0.7,
            },
        )

        assert hash1 != hash3

    def test_provider_model_list_normalization_ignores_test_context(self):
        """Provider model-list calls are shared infrastructure across tests."""
        url = "https://generativelanguage.googleapis.com/v1beta/openai/v1/models"

        first_token = set_test_context("tests/integration/inference/test_a.py::test_one")
        try:
            hash1 = normalize_inference_request("POST", url, {}, {})
        finally:
            reset_test_context(first_token)

        second_token = set_test_context("tests/integration/inference/test_b.py::test_two")
        try:
            hash2 = normalize_inference_request("POST", url, {}, {})
        finally:
            reset_test_context(second_token)

        assert hash1 == hash2

    def test_provider_model_list_normalization_matches_existing_recordings(self):
        """Shared model-list hashes preserve compatibility with existing files."""
        assert (
            normalize_inference_request("POST", "https://api.openai.com/v1/v1/models", {}, {})
            == "64a2277c90f0f42576f60c1030e3a020403d34a95f56931b792d5939f4cebc57"
        )
        assert (
            normalize_inference_request(
                "POST", "https://generativelanguage.googleapis.com/v1beta/openai/v1/models", {}, {}
            )
            == "d98e7566147f9d534bc0461f2efe61e3f525c18360a07bb3dda397579e25c27b"
        )
        assert (
            normalize_inference_request(
                "POST",
                "https://us-south.ml.cloud.ibm.com/ml/v1/v1/models",
                {},
                {"extra_query": {"project_id": "replay-mode-dummy-project"}},
            )
            == "83953d70762b611d7b1e6ff232b53374ee7c35538fb0392cf9d92f27815c4fc6"
        )

    def test_local_provider_model_list_normalization_keeps_test_context(self):
        """Local provider model-list calls return raw provider IDs and stay scoped."""
        url = "http://0.0.0.0:11434/v1/v1/models"

        first_token = set_test_context("tests/integration/inference/test_a.py::test_one")
        try:
            hash1 = normalize_inference_request("POST", url, {}, {})
        finally:
            reset_test_context(first_token)

        second_token = set_test_context("tests/integration/inference/test_b.py::test_two")
        try:
            hash2 = normalize_inference_request("POST", url, {}, {})
        finally:
            reset_test_context(second_token)

        assert hash1 != hash2

    def test_ogx_test_server_model_list_detection(self, monkeypatch):
        """Only OGX test server model-list calls bypass recording."""
        monkeypatch.setenv("TEST_API_BASE_URL", "http://localhost:8322")

        assert _is_ogx_test_server_model_list_url("http://localhost:8322/v1/v1/models")
        assert not _is_ogx_test_server_model_list_url("http://0.0.0.0:11434/v1/v1/models")
        assert not _is_ogx_test_server_model_list_url("https://api.openai.com/v1/v1/models")

    def test_request_normalization_edge_cases(self):
        """Test request normalization is precise about request content."""
        # Test that different whitespace produces different hashes (no normalization)
        hash1 = normalize_inference_request(
            "POST",
            "http://test/v1/chat/completions",
            {},
            {"messages": [{"role": "user", "content": "Hello   world\n\n"}]},
        )
        hash2 = normalize_inference_request(
            "POST", "http://test/v1/chat/completions", {}, {"messages": [{"role": "user", "content": "Hello world"}]}
        )
        assert hash1 != hash2  # Different whitespace should produce different hashes

        # Test that different float precision produces different hashes (no rounding)
        hash3 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7000001})
        hash4 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7})
        assert hash3 == hash4  # Small float precision differences should normalize to the same hash

        # String-embedded decimals with excessive precision should also normalize.
        body_with_precise_scores = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.7472640164649847",
                }
            ]
        }
        body_with_precise_scores_variation = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.74726414959878",
                }
            ]
        }
        hash5 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, body_with_precise_scores)
        hash6 = normalize_inference_request(
            "POST", "http://test/v1/chat/completions", {}, body_with_precise_scores_variation
        )
        assert hash5 == hash6

        body_with_close_scores = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.662477492560699",
                }
            ]
        }
        body_with_close_scores_variation = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.6624775971970099",
                }
            ]
        }
        hash7 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, body_with_close_scores)
        hash8 = normalize_inference_request(
            "POST", "http://test/v1/chat/completions", {}, body_with_close_scores_variation
        )
        assert hash7 == hash8

    def test_file_search_score_normalization(self):
        """Test that file_search scores are normalized for stable hashing.

        Vector search returns non-deterministic scores that vary between runs.
        The score values must be replaced entirely (not just rounded) so that
        replay hashes match regardless of the exact score returned.
        """
        url = "http://test/v1/chat/completions"

        body_a = {
            "messages": [
                {
                    "role": "tool",
                    "content": "document_id: file-123, score: 0.8523456789",
                }
            ]
        }
        body_b = {
            "messages": [
                {
                    "role": "tool",
                    "content": "document_id: file-123, score: 0.4217891234",
                }
            ]
        }
        hash_a = normalize_inference_request("POST", url, {}, body_a)
        hash_b = normalize_inference_request("POST", url, {}, body_b)
        assert hash_a == hash_b

    def test_file_search_attributes_normalization(self):
        """Test that file_search attribute dicts are stripped for stable hashing."""
        url = "http://test/v1/chat/completions"

        body_a = {
            "messages": [
                {
                    "role": "tool",
                    "content": "document_id: file-123, score: 0.85, attributes: {'document_id': 'file-123', 'source': 'a.txt'}",
                }
            ]
        }
        body_b = {
            "messages": [
                {
                    "role": "tool",
                    "content": "document_id: file-123, score: 0.85, attributes: {'document_id': 'file-456', 'source': 'b.txt'}",
                }
            ]
        }
        hash_a = normalize_inference_request("POST", url, {}, body_a)
        hash_b = normalize_inference_request("POST", url, {}, body_b)
        assert hash_a == hash_b

    def test_file_search_complete_normalization(self):
        """End-to-end test with realistic file_search results including all non-deterministic metadata.

        Tests that document_id (UUID), score, attributes, and citations all normalize correctly,
        even when document UUIDs, scores, and attributes vary completely between test runs.
        This is the realistic scenario where recordings are replayed with different IDs.
        """
        url = "http://test/v1/chat/completions"

        # First run: UUID 3ad3371d-04a9-4fa3-b837-d6dd716348e7 with score 0.0077...
        body_a = {
            "messages": [
                {
                    "role": "tool",
                    "content": (
                        "[1] document_id: 3ad3371d-04a9-4fa3-b837-d6dd716348e7, score: 0.007751386943071613, "
                        "attributes: {'region': 'us', 'category': 'engineering', 'file_id': 'file-450428750203'} "
                        "cite as <|3ad3371d-04a9-4fa3-b837-d6dd716348e7|>\n"
                        "US technical updates for Q2 2023."
                    ),
                }
            ]
        }
        # Second run: different UUID with different score and attributes
        body_b = {
            "messages": [
                {
                    "role": "tool",
                    "content": (
                        "[1] document_id: ae7bfea9-d479-4d0c-a676-babf9e6e314a, score: 0.002861692101212796, "
                        "attributes: {'region': 'us', 'category': 'marketing', 'file_id': 'file-450428750202'} "
                        "cite as <|ae7bfea9-d479-4d0c-a676-babf9e6e314a|>\n"
                        "US technical updates for Q2 2023."
                    ),
                }
            ]
        }
        hash_a = normalize_inference_request("POST", url, {}, body_a)
        hash_b = normalize_inference_request("POST", url, {}, body_b)
        assert hash_a == hash_b

    def test_non_file_search_content_still_differs(self):
        """Ensure normalization does not collapse genuinely different requests."""
        url = "http://test/v1/chat/completions"

        body_a = {"messages": [{"role": "user", "content": "What is machine learning?"}]}
        body_b = {"messages": [{"role": "user", "content": "What is deep learning?"}]}
        hash_a = normalize_inference_request("POST", url, {}, body_a)
        hash_b = normalize_inference_request("POST", url, {}, body_b)
        assert hash_a != hash_b

    def test_response_storage(self, temp_storage_dir):
        """Test the ResponseStorage class."""
        temp_storage_dir = temp_storage_dir / "test_response_storage"
        storage = ResponseStorage(temp_storage_dir)

        # Test storing and retrieving a recording
        request_hash = "test_hash_123"
        request_data = {
            "method": "POST",
            "url": "http://localhost:11434/v1/chat/completions",
            "endpoint": "/v1/chat/completions",
            "model": "llama3.2:3b",
        }
        response_data = {"body": {"content": "test response"}, "is_streaming": False}

        storage.store_recording(request_hash, request_data, response_data)

        # Verify file storage and retrieval
        retrieved = storage.find_recording(request_hash)
        assert retrieved is not None
        assert retrieved["request"]["model"] == "llama3.2:3b"
        assert retrieved["response"]["body"]["content"] == "test response"

    async def test_recording_mode(self, temp_storage_dir, real_openai_chat_response):
        """Test that recording mode captures and stores responses."""

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        temp_storage_dir = temp_storage_dir / "test_recording_mode"
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Verify the response was returned correctly
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

        # Verify recording was stored
        storage = ResponseStorage(temp_storage_dir)
        assert storage._get_test_dir().exists()

    async def test_replay_mode(self, temp_storage_dir, real_openai_chat_response):
        """Test that replay mode returns stored responses without making real calls."""

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        temp_storage_dir = temp_storage_dir / "test_replay_mode"
        # First, record a response
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

        # Now test replay mode - should not call the original method
        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_create_patch:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Verify we got the recorded response
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

                # Verify the original method was NOT called
                mock_create_patch.assert_not_called()

    async def test_record_if_missing_model_list_goes_live(self, temp_storage_dir):
        """Model lists are re-fetched live in record modes so newly pulled models are recorded."""
        from openai.types.model import Model

        def mock_list(model_ids):
            def _make(*args, **kwargs):
                async def _gen():
                    for model_id in model_ids:
                        yield Model(id=model_id, object="model", created=1, owned_by="test")

                return _gen()

            return _make

        temp_storage_dir = temp_storage_dir / "test_record_if_missing_model_list"

        # Record an initial model set
        with patch(
            "openai.resources.models.AsyncModels.list",
            side_effect=mock_list(["model-a"]),
        ):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                recorded = [m.id async for m in client.models.list()]
        assert recorded == ["model-a"]

        # A new model is now available: record-if-missing must not replay the stale
        # recorded union, it must go live and record the updated set
        with patch(
            "openai.resources.models.AsyncModels.list",
            side_effect=mock_list(["model-a", "model-b"]),
        ) as mock:
            with api_recording(mode=APIRecordingMode.RECORD_IF_MISSING, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                recorded = [m.id async for m in client.models.list()]
        assert recorded == ["model-a", "model-b"]
        assert mock.call_count == 1

        # Replay still serves the union of all recorded model sets without going live
        with patch("openai.resources.models.AsyncModels.list") as mock:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                replayed = sorted([m.id async for m in client.models.list()])
        assert replayed == ["model-a", "model-b"]
        mock.assert_not_called()

    async def test_record_if_missing_model_list_skips_write_for_unchanged_set(self, temp_storage_dir):
        """Record-if-missing must not rewrite a model-list recording when the model set is unchanged.

        vLLM embeds per-server-startup values in model objects (nested permission[].id /
        permission[].created, vLLM-specific fields such as root) that normalization does not
        cover, so unconditionally re-recording on every run produced a git diff each time and
        the commit-recordings workflow looped "Recordings update from CI" (#6635).
        """
        from openai.types.model import Model

        def mock_list(startup_marker):
            def _make(*args, **kwargs):
                async def _gen():
                    yield Model(
                        id="model-a",
                        object="model",
                        created=1700000000 + startup_marker,
                        owned_by="vllm",
                        permission=[
                            {
                                "id": f"modelperm-{startup_marker:016x}",
                                "object": "model_permission",
                                "created": 1700000000 + startup_marker,
                                "allow_create_engine": False,
                                "allow_sampling": True,
                                "allow_logprobs": True,
                                "allow_search_indices": False,
                                "allow_view_config": True,
                                "organization": "*",
                                "group": None,
                                "is_blocking": False,
                            }
                        ],
                        root=f"/models/startup-{startup_marker}",
                    )

                return _gen()

            return _make

        temp_storage_dir = temp_storage_dir / "test_rim_model_list_skip_write"

        # Record an initial model set
        with patch("openai.resources.models.AsyncModels.list", side_effect=mock_list(1)):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                recorded = [m.id async for m in client.models.list()]
        assert recorded == ["model-a"]

        first_files = sorted(temp_storage_dir.glob("**/models-*.json"))
        assert len(first_files) == 1

        # Mutate the stored file so a rewrite by the next run would be detectable
        first_files[0].write_text(first_files[0].read_text() + "// sentinel\n")

        # A fresh server start (new permission id/created, new root) serves the same
        # model set: record-if-missing must still go live, but must not rewrite the file
        with patch(
            "openai.resources.models.AsyncModels.list",
            side_effect=mock_list(2),
        ) as mock:
            with api_recording(mode=APIRecordingMode.RECORD_IF_MISSING, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                recorded = [m.id async for m in client.models.list()]
        assert recorded == ["model-a"]
        assert mock.call_count == 1
        assert len(list(temp_storage_dir.glob("**/models-*.json"))) == 1
        assert first_files[0].read_text().endswith("// sentinel\n")

        # A changed model set must still record a new file, leaving the old one untouched
        sentinel_content = first_files[0].read_text()

        def mock_list_changed(*args, **kwargs):
            async def _gen():
                yield Model(id="model-a", object="model", created=1, owned_by="vllm")
                yield Model(id="model-b", object="model", created=1, owned_by="vllm")

            return _gen()

        with patch("openai.resources.models.AsyncModels.list", side_effect=mock_list_changed) as mock:
            with api_recording(mode=APIRecordingMode.RECORD_IF_MISSING, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                recorded = sorted([m.id async for m in client.models.list()])
        assert recorded == ["model-a", "model-b"]
        assert mock.call_count == 1
        all_files = sorted(temp_storage_dir.glob("**/models-*.json"))
        assert len(all_files) == 2
        assert first_files[0].read_text() == sentinel_content

    async def test_model_list_created_timestamp_normalized(self, temp_storage_dir):
        """Model-list recordings are stable across re-records with different live timestamps.

        Providers like Ollama derive "created" from the model file's mtime in an
        ephemeral container, so a fresh record run must not produce a new diff
        for an unchanged model set (which would re-trigger CI recording runs).
        """
        from openai.types.model import Model

        def mock_list(created):
            def _make(*args, **kwargs):
                async def _gen():
                    yield Model(id="model-a", object="model", created=created, owned_by="test")

                return _gen()

            return _make

        temp_storage_dir = temp_storage_dir / "test_model_list_created_normalized"

        async def record_with(created):
            with patch("openai.resources.models.AsyncModels.list", side_effect=mock_list(created)):
                with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                    return [m.id async for m in client.models.list()]

        assert await record_with(1789759342) == ["model-a"]
        first = sorted(temp_storage_dir.glob("**/models-*.json"))
        assert len(first) == 1
        body = json.loads(first[0].read_text())["response"]["body"]
        created_values = [m["__data__"]["created"] for m in body]
        assert created_values == [0]

        # A fresh "container" reports a different mtime; the stored file must be unchanged
        assert await record_with(1789760000) == ["model-a"]
        second = sorted(temp_storage_dir.glob("**/models-*.json"))
        assert len(second) == 1
        assert second[0].read_text() == first[0].read_text()

    async def test_replay_missing_recording(self, temp_storage_dir):
        """Test that replay mode fails when no recording is found."""
        temp_storage_dir = temp_storage_dir / "test_replay_missing_recording"
        with patch("openai.resources.chat.completions.AsyncCompletions.create"):
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(RuntimeError, match="Recording not found"):
                    await client.chat.completions.create(
                        model="llama3.2:3b", messages=[{"role": "user", "content": "This was never recorded"}]
                    )

    async def test_embeddings_recording(self, temp_storage_dir, real_embeddings_response):
        """Test recording and replay of embeddings calls."""

        async def mock_create(*args, **kwargs):
            return real_embeddings_response

        temp_storage_dir = temp_storage_dir / "test_embeddings_recording"
        # Record
        with patch("openai.resources.embeddings.AsyncEmbeddings.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.embeddings.create(
                    model="nomic-embed-text", input=["Hello world", "Test embedding"]
                )

                assert len(response.data) == 2

        # Replay
        with patch("openai.resources.embeddings.AsyncEmbeddings.create") as mock_create_patch:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.embeddings.create(
                    model="nomic-embed-text", input=["Hello world", "Test embedding"]
                )

                # Verify we got the recorded response
                assert len(response.data) == 2
                assert response.data[0].embedding == [0.1, 0.2, 0.3]

                # Verify original method was not called
                mock_create_patch.assert_not_called()

    async def test_live_mode(self, real_openai_chat_response):
        """Test that live mode passes through to original methods."""

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.LIVE, storage_dir="foo"):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b", messages=[{"role": "user", "content": "Hello"}]
                )

                # Verify the response was returned
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."


class TestExceptionRecordingReplay:
    """Test that provider SDK exceptions survive the record -> replay cycle.

    Integration tests use record/replay to avoid live API calls in CI. When a
    provider raises an error (e.g. OpenAI 404), the recording system must:
    - Serialize the exception to disk during recording
    - Reconstruct the *same SDK exception type* during replay
    so that tests using ``pytest.raises(NotFoundError)`` pass in both modes.
    """

    async def test_openai_error_is_recorded_and_replayed(self, temp_storage_dir):
        """Core feature: an OpenAI error recorded during a live run is replayed identically offline."""
        # -- Setup: an OpenAI 404 error, as the SDK would raise against a real server --
        original_error = NotFoundError(
            message="Model not found",
            response=httpx.Response(
                404,
                json={"error": {"code": "not_found"}},
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            ),
            body={"error": {"code": "not_found"}},
        )
        chat_request = dict(model="test-model", messages=[{"role": "user", "content": "hi"}])
        storage = str(temp_storage_dir / "error_roundtrip")

        # -- Step 1: Record -- the error is captured to a JSON file on disk --
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create",
            side_effect=original_error,
        ):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=storage):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                with pytest.raises(NotFoundError):
                    await client.chat.completions.create(**chat_request)

        # -- Step 2: Replay -- the error is reconstructed from disk, no network call --
        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=storage):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                with pytest.raises(NotFoundError) as exc_info:
                    await client.chat.completions.create(**chat_request)

        # -- Verify: same exception type and attributes, no live call made --
        mock.assert_not_called()
        assert exc_info.value.status_code == 404
        assert exc_info.value.body == {"error": {"code": "not_found"}}

    async def test_replay_legacy_exception_format_raises_generic(self, temp_storage_dir):
        """Verify backwards compatibility with recordings made before exception_data was added.

        Older recordings store only ``exception_message`` (a plain string) without the
        structured ``exception_data`` dict. The replay path must handle this gracefully
        by raising a generic ``Exception`` with the original message, rather than
        crashing on a missing key.
        """
        temp_storage_dir = temp_storage_dir / "test_legacy_exception"
        recordings_dir = temp_storage_dir / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)

        # URL must match what AsyncOpenAI(base_url="http://localhost:11434/v1") produces
        # for /v1/chat/completions -> base_url + endpoint = .../v1/v1/chat/completions
        base_url = "http://localhost:11434/v1"
        endpoint = "/v1/chat/completions"
        url = base_url.rstrip("/") + endpoint

        request_hash = normalize_inference_request(
            "POST",
            url,
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Legacy error test"}]},
        )
        legacy_recording = {
            "test_id": None,
            "request": {
                "method": "POST",
                "url": url,
                "endpoint": endpoint,
                "body": {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Legacy error test"}]},
            },
            "response": {
                "body": None,
                "is_streaming": False,
                "is_exception": True,
                "exception_message": "Legacy formatted error",
            },
            "id_normalization_mapping": {},
        }
        with open(recordings_dir / f"{request_hash}.json", "w") as f:
            json.dump(legacy_recording, f, indent=2)

        with patch("openai.resources.chat.completions.AsyncCompletions.create"):
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(Exception) as exc_info:
                    await client.chat.completions.create(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": "Legacy error test"}],
                    )

                assert str(exc_info.value) == "Legacy formatted error"


class TestHttpxInterception:
    """The vLLM adapter's ``rerank()`` posts to a Jina-compatible ``{base_url}/rerank`` with a
    raw ``httpx.AsyncClient`` (``src/ogx/providers/remote/inference/vllm/vllm.py``) -- not
    through the OpenAI SDK, and not through aiohttp. The recorder's ``/rerank`` match used to
    live only on the aiohttp patch, so these calls were neither recorded nor replayable (#6626).
    """

    RERANK_URL = "http://vllm.test:8000/rerank"
    RERANK_PAYLOAD = {"model": "rerank-model", "query": "why", "documents": ["doc-a", "doc-b"]}
    RERANK_BODY = {
        "id": "rerank-test",
        "results": [
            {"index": 0, "relevance_score": 0.25},
            {"index": 1, "relevance_score": 0.75},
        ],
    }

    @staticmethod
    def _backend(body: dict, calls: list | None = None) -> httpx.MockTransport:
        """A stand-in backend that answers every request with ``body``."""

        def handler(request: httpx.Request) -> httpx.Response:
            if calls is not None:
                calls.append(request)
            return httpx.Response(200, json=body)

        return httpx.MockTransport(handler)

    @staticmethod
    def _no_backend() -> httpx.MockTransport:
        """A stand-in for replay CI, where no live backend is listening."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no backend is listening", request=request)

        return httpx.MockTransport(handler)

    @staticmethod
    def _recordings(storage: Path) -> list[Path]:
        return sorted(storage.rglob("*.json"))

    @pytest.mark.parametrize(
        "url,intercepted",
        [
            ("http://vllm.test:8000/rerank", True),
            ("http://vllm.test:8000/v1/rerank", True),
            ("http://ollama.test:11434/v1/messages", True),
            ("http://gemini.test/v1beta/interactions", True),
            ("http://vllm.test:8000/v1/chat/completions", False),
            ("http://vllm.test:8000/v1/embeddings", False),
        ],
    )
    def test_intercepted_paths(self, url, intercepted):
        """The raw-httpx call sites the recorder owns. Everything else -- chat/completions,
        embeddings -- is intercepted at the provider-SDK layer instead."""
        assert _should_intercept_httpx(url) is intercepted

    async def test_stream_uses_the_same_predicate_as_post(self, temp_storage_dir):
        """``post`` and ``stream`` used to carry duplicate literal path lists -- the drift that
        left ``/rerank`` on aiohttp only. Reaching ``stream`` proves both read one predicate."""
        storage = temp_storage_dir / "rerank_stream"

        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._no_backend()) as client:
                with pytest.raises(RuntimeError, match="Recording not found for httpx stream POST"):
                    async with client.stream("POST", self.RERANK_URL, json=self.RERANK_PAYLOAD):
                        pass

    async def test_rerank_post_is_recorded(self, temp_storage_dir):
        """Record mode must capture a raw-httpx ``/rerank`` response to disk."""
        storage = temp_storage_dir / "rerank_record"

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.RERANK_BODY)) as client:
                response = await client.post(self.RERANK_URL, json=self.RERANK_PAYLOAD)

        assert response.json() == self.RERANK_BODY

        recordings = self._recordings(storage)
        assert len(recordings) == 1
        recorded = json.loads(recordings[0].read_text())
        assert recorded["request"]["url"] == self.RERANK_URL
        assert recorded["request"]["payload"] == self.RERANK_PAYLOAD
        assert recorded["response"]["body"] == self.RERANK_BODY

    async def test_rerank_post_replays_with_no_backend_running(self, temp_storage_dir):
        """Replay mode must serve the recording without reaching the network -- this is the
        whole point: replay CI has no vLLM server."""
        storage = temp_storage_dir / "rerank_replay"
        calls: list[httpx.Request] = []

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(self.RERANK_BODY, calls)) as client:
                await client.post(self.RERANK_URL, json=self.RERANK_PAYLOAD)
        assert len(calls) == 1

        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._no_backend()) as client:
                replayed = await client.post(self.RERANK_URL, json=self.RERANK_PAYLOAD)

        # Still one call: replay answered from disk, the dead transport was never touched.
        assert len(calls) == 1
        # vllm.py:311-327 reads exactly these three off the response.
        assert isinstance(replayed, httpx.Response)
        assert replayed.status_code == 200
        assert replayed.json() == self.RERANK_BODY
        assert json.loads(replayed.text) == self.RERANK_BODY

    async def test_missing_rerank_recording_reports_how_to_record(self, temp_storage_dir):
        """A missing recording must fail with the recorder's own instructions, not a
        ``ConnectError`` from falling through to the network."""
        storage = temp_storage_dir / "rerank_missing"

        with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._no_backend()) as client:
                with pytest.raises(RuntimeError, match="Recording not found for httpx POST"):
                    await client.post(self.RERANK_URL, json=self.RERANK_PAYLOAD)

    async def test_messages_passthrough_is_still_intercepted(self, temp_storage_dir):
        """Regression guard: widening the predicate must not drop the Messages passthrough."""
        storage = temp_storage_dir / "messages_record"
        body = {"id": "msg-1", "content": [{"type": "text", "text": "hi"}]}
        url = "http://ollama.test:11434/v1/messages"

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend(body)) as client:
                await client.post(url, json={"model": "m", "messages": []})

        assert len(self._recordings(storage)) == 1

    async def test_unrelated_httpx_post_is_not_intercepted(self, temp_storage_dir):
        """The predicate stays narrow: chat/completions is the OpenAI SDK's job, and raw httpx
        posts elsewhere must pass straight through to the transport."""
        storage = temp_storage_dir / "unrelated"
        calls: list[httpx.Request] = []

        with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(storage)):
            async with httpx.AsyncClient(transport=self._backend({"ok": True}, calls)) as client:
                await client.post("http://vllm.test:8000/v1/chat/completions", json={"model": "m"})

        assert len(calls) == 1
        assert self._recordings(storage) == []
