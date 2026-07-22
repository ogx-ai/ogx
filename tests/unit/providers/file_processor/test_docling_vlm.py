# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from docling.models.inference_engines.vlm.base import VlmEngineInput
from PIL import Image

from ogx.providers.inline.file_processor.docling.config import DoclingFileProcessorConfig
from ogx.providers.inline.file_processor.docling.vlm_engine import OgxInferenceVlmEngine


def _make_chat_response(text: str, finish_reason: str = "stop"):
    """Build a minimal chat completion response object."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _make_inference_api(response_text: str = "extracted text", finish_reason: str = "stop"):
    """Build a mock inference API that returns a canned response."""
    api = MagicMock()
    api.openai_chat_completion = AsyncMock(
        return_value=_make_chat_response(response_text, finish_reason),
    )
    return api


def _make_vlm_input(prompt: str = "Describe this image.") -> VlmEngineInput:
    """Build a VlmEngineInput with a small test image."""
    image = Image.new("RGB", (10, 10), color="red")
    return VlmEngineInput(
        image=image,
        prompt=prompt,
        temperature=0.0,
        max_new_tokens=256,
        stop_strings=[],
    )


class TestOgxInferenceVlmEngine:
    def setup_method(self):
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._loop_thread.start()
        self.inference_api = _make_inference_api()
        self.engine = OgxInferenceVlmEngine(
            inference_api=self.inference_api,
            model="granite-docling-258M",
            event_loop=self.loop,
        )

    def teardown_method(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._loop_thread.join(timeout=5)
        self.loop.close()

    def test_predict_batch_empty(self):
        assert self.engine.predict_batch([]) == []

    def test_predict_single_returns_text(self):
        inference_api = _make_inference_api("hello world")
        engine = OgxInferenceVlmEngine(
            inference_api=inference_api,
            model="granite-docling-258M",
            event_loop=self.loop,
        )

        results = engine.predict_batch([_make_vlm_input()])

        assert len(results) == 1
        assert results[0].text == "hello world"
        assert results[0].stop_reason == "end_of_sequence"

    def test_predict_single_passes_correct_model(self):
        self.engine.predict_batch([_make_vlm_input()])

        call_args = self.inference_api.openai_chat_completion.call_args
        request = call_args[0][0]
        assert request.model == "granite-docling-258M"

    def test_predict_single_sends_image_and_prompt(self):
        self.engine.predict_batch([_make_vlm_input("Extract tables.")])

        call_args = self.inference_api.openai_chat_completion.call_args
        request = call_args[0][0]
        messages = request.messages
        assert len(messages) == 1

        content = messages[0].content
        assert len(content) == 2
        assert content[0].image_url.url.startswith("data:image/png;base64,")
        assert content[1].text == "Extract tables."

    def test_predict_single_sets_temperature_and_max_tokens(self):
        inp = VlmEngineInput(
            image=Image.new("RGB", (10, 10)),
            prompt="test",
            temperature=0.5,
            max_new_tokens=128,
            stop_strings=["<end>"],
        )
        self.engine.predict_batch([inp])

        request = self.inference_api.openai_chat_completion.call_args[0][0]
        assert request.temperature == 0.5
        assert request.max_tokens == 128
        assert request.stop == ["<end>"]
        assert request.stream is False

    def test_predict_single_length_stop_reason(self):
        inference_api = _make_inference_api("truncated", finish_reason="length")
        engine = OgxInferenceVlmEngine(
            inference_api=inference_api,
            model="test-model",
            event_loop=self.loop,
        )

        results = engine.predict_batch([_make_vlm_input()])
        assert results[0].stop_reason == "length"

    def test_predict_single_content_filter_stop_reason(self):
        inference_api = _make_inference_api("", finish_reason="content_filter")
        engine = OgxInferenceVlmEngine(
            inference_api=inference_api,
            model="test-model",
            event_loop=self.loop,
        )

        results = engine.predict_batch([_make_vlm_input()])
        assert results[0].stop_reason == "content_filtered"

    def test_predict_single_includes_metadata(self):
        results = self.engine.predict_batch([_make_vlm_input()])
        assert results[0].metadata["model"] == "granite-docling-258M"
        assert "generation_time" in results[0].metadata

    def test_predict_batch_multiple_inputs(self):
        inputs = [_make_vlm_input(f"prompt {i}") for i in range(3)]
        results = self.engine.predict_batch(inputs)

        assert len(results) == 3
        assert self.inference_api.openai_chat_completion.call_count == 3

    def test_predict_single_inference_error_returns_error_output(self):
        inference_api = _make_inference_api()
        inference_api.openai_chat_completion = AsyncMock(
            side_effect=RuntimeError("connection refused"),
        )
        engine = OgxInferenceVlmEngine(
            inference_api=inference_api,
            model="test-model",
            event_loop=self.loop,
        )

        results = engine.predict_batch([_make_vlm_input()])
        assert len(results) == 1
        assert results[0].text == ""
        assert results[0].stop_reason == "error"


class TestDoclingFileProcessorVlmConfig:
    def test_vlm_model_not_set_uses_standard_pipeline(self):
        config = DoclingFileProcessorConfig()
        assert config.vlm_model is None
        assert config.vlm_preset == "granite_docling"

    def test_missing_inference_api_falls_back(self):
        """When vlm_model is set but no inference API, should fall back to standard pipeline."""
        try:
            from ogx.providers.inline.file_processor.docling.docling import DoclingFileProcessor
        except ImportError:
            pytest.skip("docling not fully installed")

        config = DoclingFileProcessorConfig(vlm_model="some-model")
        processor = DoclingFileProcessor(config, files_api=MagicMock(), inference_api=None)
        assert processor._vlm_enabled is False

    def test_invalid_preset_raises_error(self):
        """An invalid vlm_preset should raise ValueError at construction."""
        try:
            from ogx.providers.inline.file_processor.docling.docling import DoclingFileProcessor
        except ImportError:
            pytest.skip("docling not fully installed")

        config = DoclingFileProcessorConfig(vlm_model="some-model", vlm_preset="nonexistent_preset")
        with pytest.raises(ValueError, match="Invalid vlm_preset"):
            DoclingFileProcessor(config, files_api=MagicMock(), inference_api=MagicMock())
