# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast


def _install_google_shims() -> None:
    """Install Google SDK shim modules for tests."""
    google_module = sys.modules.get("google")
    if google_module is None:
        google_module = ModuleType("google")
        google_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["google"] = google_module

    genai_module = ModuleType("google.genai")

    class MockClient:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.kwargs = kwargs
            self.models = SimpleNamespace(list=lambda: [])
            self.aio = SimpleNamespace(models=SimpleNamespace())

    cast(Any, genai_module).Client = MockClient
    sys.modules["google.genai"] = genai_module
    cast(Any, google_module).genai = genai_module

    genai_types_module = ModuleType("google.genai.types")

    class FunctionCallingConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class ToolConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class Tool:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class ThinkingConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            if "thinking_config" in kwargs and isinstance(kwargs["thinking_config"], dict):
                kwargs["thinking_config"] = ThinkingConfig(**kwargs["thinking_config"])
            self.__dict__.update(kwargs)

    class ListModelsConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class HttpOptions:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    class EmbedContentConfig:
        def __init__(self, **kwargs):
            """Initialize shim attributes from kwargs."""
            self.__dict__.update(kwargs)

    cast(Any, genai_types_module).FunctionCallingConfig = FunctionCallingConfig
    cast(Any, genai_types_module).ToolConfig = ToolConfig
    cast(Any, genai_types_module).Tool = Tool
    cast(Any, genai_types_module).GenerateContentConfig = GenerateContentConfig
    cast(Any, genai_types_module).ThinkingConfig = ThinkingConfig
    cast(Any, genai_types_module).ListModelsConfig = ListModelsConfig
    cast(Any, genai_types_module).HttpOptions = HttpOptions
    cast(Any, genai_types_module).EmbedContentConfig = EmbedContentConfig

    sys.modules["google.genai.types"] = genai_types_module
    cast(Any, genai_module).types = genai_types_module

    oauth2_module = ModuleType("google.oauth2")
    sys.modules["google.oauth2"] = oauth2_module
    cast(Any, google_module).oauth2 = oauth2_module

    credentials_module = ModuleType("google.oauth2.credentials")

    class Credentials:
        def __init__(self, token: str):
            """Initialize shim attributes from kwargs."""
            self.token = token

    cast(Any, credentials_module).Credentials = Credentials
    sys.modules["google.oauth2.credentials"] = credentials_module
    cast(Any, oauth2_module).credentials = credentials_module


_install_google_shims()

# ── shared fixtures and helpers for test_adapter_*.py ──

from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402

from ogx.providers.remote.inference.vertexai.config import VertexAIConfig  # noqa: E402
from ogx.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter  # noqa: E402


async def _async_pager(items):
    """Yield items through an async iterator."""
    for item in items:
        yield item


def _make_fake_streaming_chunk(text: str = "chunk") -> SimpleNamespace:
    """Build a fake streaming chunk object."""
    part = SimpleNamespace(text=text, function_call=None, thought=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP", index=0, logprobs_result=None)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


@pytest.fixture
def vertex_config() -> VertexAIConfig:
    """Handle vertex config."""
    return VertexAIConfig(project="test-project", location="global")


@pytest.fixture
def adapter(vertex_config: VertexAIConfig) -> VertexAIInferenceAdapter:
    """Handle adapter."""
    a = VertexAIInferenceAdapter(config=vertex_config)
    cast(Any, a).__provider_id__ = "vertexai"
    return a


@pytest.fixture
def patch_chat_completion_dependencies(monkeypatch):
    """Patch chat completion dependencies for the test."""

    def factory(
        adapter: VertexAIInferenceAdapter,
        *,
        capture_messages: bool = False,
        capture_tools: bool = False,
        capture_generation_kwargs: bool = False,
    ) -> dict[str, Any]:
        """Create an adapter with a mocked client."""
        fake_response = object()
        fake_completion = object()
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=fake_response)))
        )
        captured: dict[str, Any] = {"fake_completion": fake_completion}

        async def _provider_model_id(_: str) -> str:
            """Return a fixed provider model identifier."""
            return "gemini-2.5-flash"

        monkeypatch.setattr(adapter, "_get_provider_model_id", _provider_model_id)
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        if capture_generation_kwargs:

            def _build_generation_config(*_args, **kwargs):
                """Build generation config."""
                captured["build_generation_config_kwargs"] = kwargs
                return object()

            monkeypatch.setattr(adapter, "_build_generation_config", _build_generation_config)
        else:
            monkeypatch.setattr(adapter, "_build_generation_config", lambda *_args, **_kwargs: object())

        if capture_messages:

            def _convert_messages(messages):
                """Convert messages."""
                captured["messages"] = messages
                return None, [{"role": "user", "parts": [{"text": "ok"}]}]

        else:

            def _convert_messages(messages):
                """Convert messages."""
                return None, [{"role": "user", "parts": [{"text": "ok"}]}]

        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.converters.convert_openai_messages_to_gemini",
            _convert_messages,
        )

        if capture_tools:

            def _convert_tools_capture(tools):
                """Convert tools capture."""
                captured["tools_passed"] = tools
                return None

            convert_tools = _convert_tools_capture
        else:

            def _convert_tools_passthrough(_tools):
                """Convert tools passthrough."""
                return None

            convert_tools = _convert_tools_passthrough

        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.converters.convert_openai_tools_to_gemini",
            convert_tools,
        )
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.converters.convert_gemini_response_to_openai",
            lambda response, model: fake_completion,
        )

        return captured

    return factory


@pytest.fixture
def make_adapter_with_mock_embed(monkeypatch):
    """Create adapter with mock embed."""

    def factory(
        embeddings: list[list[float]], capture: dict | None = None, usage_metadata: SimpleNamespace | None = None
    ):
        """Create an adapter with a mocked client."""
        a = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        cast(Any, a).__provider_id__ = "vertexai"

        fake_embedding_objects = [SimpleNamespace(value=vec) for vec in embeddings]
        fake_response = SimpleNamespace(embeddings=fake_embedding_objects)
        if usage_metadata is not None:
            fake_response.usage_metadata = usage_metadata

        embed_mock = AsyncMock(return_value=fake_response)

        if capture is not None:

            async def capturing_embed(*args, **kwargs):
                """Handle capturing embed."""
                capture.update(kwargs)
                return fake_response

            embed_mock = capturing_embed

        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_mock)))
        monkeypatch.setattr(a, "_get_client", lambda: fake_client)
        return a, embed_mock

    return factory
