# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

import torch

from ogx.log import get_logger
from ogx_api import (
    InferenceProvider,
    Model,
    ModelsProtocolPrivate,
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    RerankData,
    RerankResponse,
)
from ogx_api.inference import RerankRequest

from .config import TransformersInferenceConfig

# key is model name, value is tuple of (AutoTokenizer, AutoModelForCausalLM)
RERANKER_MODELS: dict[str, tuple] = {}

RERANKER_MODELS_LOCK: asyncio.Lock = asyncio.Lock()

DEFAULT_RERANKER_INSTRUCTION = "Given the search query, retrieve relevant passages that answer the query"

log = get_logger(name=__name__, category="inference")


class TransformersInferenceImpl(
    InferenceProvider,
    ModelsProtocolPrivate,
):
    """Inference provider for neural reranking using HuggingFace transformers models."""

    __provider_id__: str

    def __init__(self, config: TransformersInferenceConfig) -> None:
        self.config = config

    async def openai_chat_completions_with_reasoning(self, params: OpenAIChatCompletionRequestWithExtraBody) -> None:
        raise NotImplementedError("Transformers provider does not support reasoning in chat completions")

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def should_refresh_models(self) -> bool:
        return False

    async def list_models(self) -> list[Model] | None:
        return [
            Model(
                identifier="Qwen/Qwen3-Reranker-0.6B",
                provider_resource_id="Qwen/Qwen3-Reranker-0.6B",
                provider_id=self.__provider_id__,
                model_type=ModelType.rerank,
            ),
        ]

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("OpenAI completion not supported by transformers provider")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        raise NotImplementedError("OpenAI chat completion not supported by transformers provider")

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError("OpenAI embeddings not supported by transformers provider")

    async def rerank(
        self,
        request: RerankRequest,
    ) -> RerankResponse:
        """
        Rerank documents based on query relevance using reranker model
        """
        if not request.items:
            return RerankResponse(data=[])

        if request.max_num_results is not None and request.max_num_results < 1:
            raise ValueError(f"max_num_results must be >= 1, got {request.max_num_results}")

        # Get the tokenizer and reranker model
        reranker_tokenizer, reranker_model = await self.load_reranker_model(request.model)

        query_text = self.extract_text(request.query)
        item_texts = [self.extract_text(item) for item in request.items]

        # Build formatted instruction pairs for each query-document combination
        pairs = [self.format_instruction(DEFAULT_RERANKER_INSTRUCTION, query_text, doc) for doc in item_texts]

        # Compute relevance scores
        relevance_scores = await asyncio.to_thread(
            self.compute_reranked_scores, reranker_tokenizer, reranker_model, pairs
        )

        # Sort relevance scores in descending order
        indexed_scores = [(i, score) for i, score in enumerate(relevance_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if request.max_num_results is not None:
            indexed_scores = indexed_scores[: request.max_num_results]

        rerank_data = [RerankData(index=idx, relevance_score=score) for idx, score in indexed_scores]

        return RerankResponse(data=rerank_data)

    async def load_reranker_model(self, model: str) -> tuple[Any, Any]:
        cached = RERANKER_MODELS.get(model)
        if cached is not None:
            return cached

        # Prevents multiple concurrent requests from loading the same model in memory simultaneously
        async with RERANKER_MODELS_LOCK:
            cached = RERANKER_MODELS.get(model)
            if cached is not None:
                return cached

            log.info(f"Loading reranker model {model}...")

            def load_model():
                from transformers import AutoModelForCausalLM, AutoTokenizer

                if threading.current_thread() is not threading.main_thread():
                    # PyTorch's OpenMP kernels can segfault when spawned from background
                    # threads with the default parallel settings, so force a single-threaded CPU run.
                    log.debug("Constraining torch threads to 1 (running in worker thread)")
                    torch.set_num_threads(1)

                # Load reranker model for reranking
                reranker_tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
                reranker_model = AutoModelForCausalLM.from_pretrained(model).eval()

                return reranker_tokenizer, reranker_model

            loaded_tokenizer, loaded_model = await asyncio.to_thread(load_model)
            RERANKER_MODELS[model] = (loaded_tokenizer, loaded_model)
            return loaded_tokenizer, loaded_model

    @torch.no_grad()
    def compute_reranked_scores(
        self,
        reranker_tokenizer: Any,
        reranker_model: Any,
        pairs: list[str],
    ) -> list[float]:
        """Compute relevance scores using reranker.

        Args:
            reranker_tokenizer: tokenizer
            reranker_model: reranker
            pairs: list of strings where each string contains instruct, query and the document

        Returns:
            List of scores
        """
        # Reranker configuration
        max_length = 8192
        token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
        token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

        # Tokenize pairs
        inputs = reranker_tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )

        for i, tokens in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + tokens + suffix_tokens

        inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

        for key in inputs:
            inputs[key] = inputs[key].to(reranker_model.device)

        batch_scores = reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores: list[float] = batch_scores[:, 1].exp().tolist()
        return scores

    def extract_text(
        self, value: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam
    ) -> str:
        """Extract plain text from a query or item value."""
        if isinstance(value, str):
            return value
        if isinstance(value, OpenAIChatCompletionContentPartTextParam):
            return value.text
        raise ValueError(f"Unsupported content type for reranking: {type(value)}. Only text is supported.")

    def format_instruction(self, instruction: str, query: str, document: str) -> str:
        """Format a query-document pair with instruction for the reranker model.

        Args:
            instruction: instruction for reranker model
            query: original query for retrieval
            document: retrieved document

        Returns:
            The string that contains query-document pair
        """
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
