# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import platform
import struct
from typing import TYPE_CHECKING

from ogx.log import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from ogx_api import (
    ModelStore,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    validate_embeddings_input_is_text,
)

EMBEDDING_MODELS: dict[str, "SentenceTransformer"] = {}
EMBEDDING_MODELS_LOCK = asyncio.Lock()

DARWIN = "Darwin"


log = get_logger(name=__name__, category="providers::utils")


class SentenceTransformerEmbeddingMixin:
    """Mixin providing OpenAI-compatible embeddings via sentence-transformers models."""

    model_store: ModelStore

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        # Validate that input contains only text, not token arrays
        validate_embeddings_input_is_text(params)

        # Convert input to list format if it's a single string
        input_list = [params.input] if isinstance(params.input, str) else params.input
        if not input_list:
            raise ValueError("Empty list not supported")

        # Get trust_remote_code setting from config
        trust_remote_code = getattr(self.config, "trust_remote_code", False)

        # Get the model and generate embeddings
        embedding_model = await self._load_sentence_transformer_model(params.model, trust_remote_code)
        embeddings = await asyncio.to_thread(embedding_model.encode, input_list, show_progress_bar=False)

        # Convert embeddings to the requested format
        data = []
        for i, embedding in enumerate(embeddings):
            if params.encoding_format == "base64":
                # Convert float array to base64 string
                float_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                embedding_value = base64.b64encode(float_bytes).decode("ascii")
            else:
                # Default to float format
                embedding_value = embedding.tolist()

            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_value,
                    index=i,
                )
            )

        # Not returning actual token usage
        usage = OpenAIEmbeddingUsage(prompt_tokens=-1, total_tokens=-1)
        return OpenAIEmbeddingsResponse(
            data=data,
            model=params.model,
            usage=usage,
        )

    async def _load_sentence_transformer_model(
        self, model: str, trust_remote_code: bool = False
    ) -> "SentenceTransformer":
        loaded_model = EMBEDDING_MODELS.get(model)
        if loaded_model is not None:
            return loaded_model

        async with EMBEDDING_MODELS_LOCK:
            loaded_model = EMBEDDING_MODELS.get(model)
            if loaded_model is not None:
                return loaded_model

            log.info(f"Loading sentence transformer for {model}...")

            def _load_model():
                import torch
                from sentence_transformers import SentenceTransformer

                platform_name = platform.system()
                if platform_name == DARWIN:
                    # PyTorch's OpenMP kernels can segfault on macOS when spawned from background
                    # threads with the default parallel settings, so force a single-threaded CPU run.
                    log.debug(f"Constraining torch threads on {platform_name} to a single worker")
                    torch.set_num_threads(1)

                return SentenceTransformer(model, trust_remote_code=trust_remote_code)

            loaded_model = await asyncio.to_thread(_load_model)
            EMBEDDING_MODELS[model] = loaded_model
            return loaded_model
