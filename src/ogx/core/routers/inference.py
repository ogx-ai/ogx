# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Body
from openai.types.chat import ChatCompletionToolChoiceOptionParam as OpenAIChatCompletionToolChoiceOptionParam
from openai.types.chat import ChatCompletionToolParam as OpenAIChatCompletionToolParam
from pydantic import TypeAdapter

from ogx.core.access_control.access_control import is_action_allowed
from ogx.core.datatypes import ModelWithOwner
from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx.telemetry.inference_metrics import (
    create_inference_metric_attributes,
    inference_duration,
    inference_time_to_first_token,
    inference_tokens_per_second,
)
from ogx_api import (
    GetChatCompletionRequest,
    HealthResponse,
    HealthStatus,
    Inference,
    ListChatCompletionsRequest,
    ListOpenAIChatCompletionResponse,
    ModelNotFoundError,
    ModelType,
    ModelTypeError,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionWithReasoning,
    OpenAIChoice,
    OpenAIChoiceLogprobs,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAICompletionWithInputMessages,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIMessageParam,
    OpenAITokenLogProb,
    OpenAITopLogProb,
    RegisterModelRequest,
    RerankResponse,
    RoutingTable,
)
from ogx_api.inference.models import RerankRequest

logger = get_logger(name=__name__, category="core::routers")


class InferenceRouter(Inference):
    """Routes to an provider based on the model"""

    def __init__(
        self,
        routing_table: RoutingTable,
        store: InferenceStore | None = None,
    ) -> None:
        logger.debug("Initializing InferenceRouter")
        self.routing_table = routing_table
        self.store = store

    async def initialize(self) -> None:
        logger.debug("InferenceRouter.initialize")

    async def shutdown(self) -> None:
        logger.debug("InferenceRouter.shutdown")
        if self.store:
            try:
                await self.store.shutdown()
            except Exception as e:
                logger.warning("Error during InferenceStore shutdown", error=str(e))

    async def register_model(
        self,
        model_id: str,
        provider_model_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_type: ModelType | None = None,
    ) -> None:
        logger.debug(
            "InferenceRouter.register_model",
            model_id=model_id,
            provider_model_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
        )
        request = RegisterModelRequest(
            model_id=model_id,
            provider_model_id=provider_model_id,
            provider_id=provider_id,
            metadata=metadata,
            model_type=model_type,
        )
        await self.routing_table.register_model(request)

    async def _get_model_provider(self, model_id: str, expected_model_type: str) -> tuple[Inference, str]:
        model = await self.routing_table.get_object_by_identifier("model", model_id)
        if model:
            if model.model_type != expected_model_type:
                raise ModelTypeError(model_id, model.model_type, expected_model_type)

            provider = await self.routing_table.get_provider_impl(model.identifier)
            return provider, model.provider_resource_id

        # Handles cases where clients use the provider format directly
        return await self._get_provider_by_fallback(model_id, expected_model_type)

    async def _get_provider_by_fallback(self, model_id: str, expected_model_type: str) -> tuple[Inference, str]:
        """
        Handle fallback case where model_id is in provider_id/provider_resource_id format.
        """
        splits = model_id.split("/", maxsplit=1)
        if len(splits) != 2:
            raise ModelNotFoundError(model_id)

        provider_id, provider_resource_id = splits

        # Check if provider exists
        if provider_id not in self.routing_table.impls_by_provider_id:
            logger.warning("Provider not found for model", provider_id=provider_id, model_id=model_id)
            raise ModelNotFoundError(model_id)

        # Create a temporary model object for RBAC check
        temp_model = ModelWithOwner(
            identifier=model_id,
            provider_id=provider_id,
            provider_resource_id=provider_resource_id,
            model_type=expected_model_type,
            metadata={},  # Empty metadata for temporary object
        )

        # Perform RBAC check
        user = get_authenticated_user()
        if not is_action_allowed(self.routing_table.policy, "read", temp_model, user):
            logger.debug(
                "Access denied to model via fallback path for user",
                model_id=model_id,
                user=user.principal if user else "anonymous",
            )
            raise ModelNotFoundError(model_id)

        return self.routing_table.impls_by_provider_id[provider_id], provider_resource_id

    async def rerank(
        self,
        params: RerankRequest,
    ) -> RerankResponse:
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.rerank)
        params.model = provider_resource_id
        return await provider.rerank(params)

    async def openai_completion(
        self,
        params: Annotated[OpenAICompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAICompletion:
        logger.debug(
            "InferenceRouter.openai_completion: model=, stream=, prompt",
            model=params.model,
            stream=params.stream,
            prompt=params.prompt,
        )
        request_model_id = params.model
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.llm)
        params.model = provider_resource_id

        if params.stream:
            return await provider.openai_completion(params)

        response = await provider.openai_completion(params)
        response.model = request_model_id
        return response

    async def openai_chat_completion(
        self,
        params: Annotated[OpenAIChatCompletionRequestWithExtraBody, Body(...)],
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        logger.debug(
            "InferenceRouter.openai_chat_completion: model=, stream=, messages",
            model=params.model,
            stream=params.stream,
            messages=params.messages,
        )
        request_model_id = params.model
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.llm)
        params.model = provider_resource_id

        # Use the OpenAI client for a bit of extra input validation without
        # exposing the OpenAI client itself as part of our API surface
        if params.tool_choice:
            TypeAdapter(OpenAIChatCompletionToolChoiceOptionParam).validate_python(params.tool_choice)
            if params.tools is None:
                raise ValueError("'tool_choice' is only allowed when 'tools' is also provided")
        if params.tools:
            for tool in params.tools:
                TypeAdapter(OpenAIChatCompletionToolParam).validate_python(tool)

        # Some providers make tool calls even when tool_choice is "none"
        # so just clear them both out to avoid unexpected tool calls
        if params.tool_choice == "none" and params.tools is not None:
            params.tool_choice = None
            params.tools = None

        if params.stream:
            response_stream = await provider.openai_chat_completion(params)

            # For streaming, the provider returns AsyncIterator[OpenAIChatCompletionChunk]
            # We need to add metrics to each chunk and store the final completion
            return self.stream_tokens_and_compute_metrics_openai_chat(
                response=response_stream,
                fully_qualified_model_id=request_model_id,
                provider_id=provider.__provider_id__,
                messages=params.messages,
            )

        start_time = time.perf_counter()
        status = "success"
        try:
            response = await self._nonstream_openai_chat_completion(provider, params)
        except asyncio.CancelledError:
            status = "error"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            attrs = create_inference_metric_attributes(
                model=request_model_id,
                provider=provider.__provider_id__,
                stream=False,
                status=status,
            )
            inference_duration.record(duration, attributes=attrs)

        response.model = request_model_id

        if response.usage and response.usage.completion_tokens and duration > 0:
            tokens_per_sec = response.usage.completion_tokens / duration
            inference_tokens_per_second.record(
                tokens_per_sec,
                attributes=create_inference_metric_attributes(
                    model=request_model_id,
                    provider=provider.__provider_id__,
                    stream=False,
                    status="success",
                ),
            )

        # Store the response with the ID that will be returned to the client
        if self.store:
            asyncio.create_task(self.store.store_chat_completion(response, params.messages))

        return response

    async def openai_chat_completions_with_reasoning(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletionWithReasoning | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]:
        """Called by the Responses layer when a user requests reasoning.

        Routes to the provider's reasoning-aware CC implementation, which
        handles mapping reasoning fields to/from the provider's format.
        If the provider doesn't support reasoning, raises an error.
        """
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.llm)
        params.model = provider_resource_id
        # Not all providers implement openai_chat_completions_with_reasoning.
        # the Responses layer catches them and falls back to regular CC
        # with a warning.
        return await provider.openai_chat_completions_with_reasoning(params)

    async def openai_embeddings(
        self,
        params: Annotated[OpenAIEmbeddingsRequestWithExtraBody, Body(...)],
    ) -> OpenAIEmbeddingsResponse:
        logger.debug(
            "InferenceRouter.openai_embeddings: model=, input_type=, encoding_format=, dimensions",
            model=params.model,
            type_params_input=type(params.input),
            encoding_format=params.encoding_format,
            dimensions=params.dimensions,
        )
        request_model_id = params.model
        provider, provider_resource_id = await self._get_model_provider(params.model, ModelType.embedding)
        params.model = provider_resource_id

        response = await provider.openai_embeddings(params)
        response.model = request_model_id
        return response

    async def list_chat_completions(
        self,
        request: ListChatCompletionsRequest,
    ) -> ListOpenAIChatCompletionResponse:
        if self.store:
            return await self.store.list_chat_completions(
                after=request.after,
                limit=request.limit,
                model=request.model,
                order=request.order,
            )
        raise NotImplementedError("List chat completions is not supported: inference store is not configured.")

    async def get_chat_completion(self, request: GetChatCompletionRequest) -> OpenAICompletionWithInputMessages:
        if self.store:
            return await self.store.get_chat_completion(request.completion_id)
        raise NotImplementedError("Get chat completion is not supported: inference store is not configured.")

    async def _nonstream_openai_chat_completion(
        self, provider: Inference, params: OpenAIChatCompletionRequestWithExtraBody
    ) -> OpenAIChatCompletion:
        response = await provider.openai_chat_completion(params)
        for choice in response.choices:
            # some providers return an empty list for no tool calls in non-streaming responses
            # but the OpenAI API returns None. So, set tool_calls to None if it's empty
            if choice.message and choice.message.tool_calls is not None and len(choice.message.tool_calls) == 0:
                choice.message.tool_calls = None
        return response

    async def health(self) -> dict[str, HealthResponse]:
        health_statuses = {}
        timeout = 1  # increasing the timeout to 1 second for health checks
        for provider_id, impl in self.routing_table.impls_by_provider_id.items():
            try:
                # check if the provider has a health method
                if not hasattr(impl, "health"):
                    continue
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                health_statuses[provider_id] = health
            except TimeoutError:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR,
                    message=f"Health check timed out after {timeout} seconds",
                )
            except NotImplementedError:
                health_statuses[provider_id] = HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
            except Exception as e:
                health_statuses[provider_id] = HealthResponse(
                    status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"
                )
        return health_statuses

    async def stream_tokens_and_compute_metrics_openai_chat(
        self,
        response: AsyncIterator[OpenAIChatCompletionChunk],
        fully_qualified_model_id: str,
        provider_id: str,
        messages: list[OpenAIMessageParam] | None = None,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        """Stream OpenAI chat completion chunks, compute metrics, and store the final completion."""
        id = None
        created = None
        choices_data: dict[int, dict[str, Any]] = {}
        start_time = time.perf_counter()
        first_token_time: float | None = None
        completion_tokens: int | None = None
        status = "success"

        try:
            async for chunk in response:
                # Skip None chunks
                if chunk is None:
                    continue

                # Capture ID and created timestamp from first chunk
                if id is None and chunk.id:
                    id = chunk.id
                if created is None and chunk.created:
                    created = chunk.created

                chunk.model = fully_qualified_model_id

                # Accumulate choice data for final assembly
                if chunk.choices:
                    for choice_delta in chunk.choices:
                        idx = choice_delta.index
                        if idx not in choices_data:
                            choices_data[idx] = {
                                "content_parts": [],
                                "tool_calls_builder": {},
                                "finish_reason": "stop",
                                "logprobs_content_parts": [],
                            }
                        current_choice_data = choices_data[idx]

                        if choice_delta.delta:
                            delta = choice_delta.delta
                            if delta.content:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                current_choice_data["content_parts"].append(delta.content)
                            if delta.tool_calls:
                                for tool_call_delta in delta.tool_calls:
                                    tc_idx = tool_call_delta.index
                                    if tc_idx not in current_choice_data["tool_calls_builder"]:
                                        current_choice_data["tool_calls_builder"][tc_idx] = {
                                            "id": None,
                                            "type": "function",
                                            "function_name_parts": [],
                                            "function_arguments_parts": [],
                                        }
                                    builder = current_choice_data["tool_calls_builder"][tc_idx]
                                    if tool_call_delta.id:
                                        builder["id"] = tool_call_delta.id
                                    if tool_call_delta.type:
                                        builder["type"] = tool_call_delta.type
                                    if tool_call_delta.function:
                                        if tool_call_delta.function.name:
                                            builder["function_name_parts"].append(tool_call_delta.function.name)
                                        if tool_call_delta.function.arguments:
                                            builder["function_arguments_parts"].append(
                                                tool_call_delta.function.arguments
                                            )
                        if choice_delta.finish_reason:
                            current_choice_data["finish_reason"] = choice_delta.finish_reason

                        # Convert logprobs from chat completion format to responses format
                        # Chat completion returns list of ChatCompletionTokenLogprob, but
                        # expecting list of OpenAITokenLogProb in OpenAIChoice
                        if choice_delta.logprobs and choice_delta.logprobs.content:
                            converted_logprobs = []
                            for token_logprob in choice_delta.logprobs.content:
                                top_logprobs = None
                                if token_logprob.top_logprobs:
                                    top_logprobs = [
                                        OpenAITopLogProb(
                                            token=tlp.token,
                                            bytes=tlp.bytes,
                                            logprob=tlp.logprob,
                                        )
                                        for tlp in token_logprob.top_logprobs
                                    ]
                                converted_logprobs.append(
                                    OpenAITokenLogProb(
                                        token=token_logprob.token,
                                        bytes=token_logprob.bytes,
                                        logprob=token_logprob.logprob,
                                        top_logprobs=top_logprobs,
                                    )
                                )
                            # Update choice delta with the newly formatted logprobs object
                            choice_delta.logprobs.content = converted_logprobs
                            current_choice_data["logprobs_content_parts"].extend(converted_logprobs)

                # Capture usage from the final chunk (providers send usage in the last chunk)
                if chunk.usage and chunk.usage.completion_tokens:
                    completion_tokens = chunk.usage.completion_tokens

                # Compute metrics on final chunk
                if chunk.choices and chunk.choices[0].finish_reason:
                    completion_text = ""
                    for choice_data in choices_data.values():
                        completion_text += "".join(choice_data["content_parts"])

                yield chunk
        except asyncio.CancelledError:
            status = "error"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            attrs = create_inference_metric_attributes(
                model=fully_qualified_model_id,
                provider=provider_id,
                stream=True,
                status=status,
            )
            inference_duration.record(duration, attributes=attrs)

            if first_token_time is not None:
                ttft = first_token_time - start_time
                inference_time_to_first_token.record(ttft, attributes=attrs)

            if completion_tokens and duration > 0:
                tokens_per_sec = completion_tokens / duration
                inference_tokens_per_second.record(tokens_per_sec, attributes=attrs)

            # Store the final assembled completion
            if id and self.store and messages:
                assembled_choices: list[OpenAIChoice] = []
                for choice_idx, choice_data in choices_data.items():
                    content_str = "".join(choice_data["content_parts"])
                    assembled_tool_calls: list[OpenAIChatCompletionToolCall] = []
                    if choice_data["tool_calls_builder"]:
                        for tc_build_data in choice_data["tool_calls_builder"].values():
                            if tc_build_data["id"]:
                                func_name = "".join(tc_build_data["function_name_parts"])
                                func_args = "".join(tc_build_data["function_arguments_parts"])
                                assembled_tool_calls.append(
                                    OpenAIChatCompletionToolCall(
                                        id=tc_build_data["id"],
                                        type=tc_build_data["type"],
                                        function=OpenAIChatCompletionToolCallFunction(
                                            name=func_name, arguments=func_args
                                        ),
                                    )
                                )
                    message = OpenAIChatCompletionResponseMessage(
                        role="assistant",
                        content=content_str if content_str else None,
                        tool_calls=assembled_tool_calls if assembled_tool_calls else None,
                    )
                    logprobs_content = choice_data["logprobs_content_parts"]
                    final_logprobs = OpenAIChoiceLogprobs(content=logprobs_content) if logprobs_content else None

                    assembled_choices.append(
                        OpenAIChoice(
                            finish_reason=choice_data["finish_reason"],
                            index=choice_idx,
                            message=message,
                            logprobs=final_logprobs,
                        )
                    )

                final_response = OpenAIChatCompletion(
                    id=id,
                    choices=assembled_choices,
                    created=created or int(time.time()),
                    model=fully_qualified_model_id,
                    object="chat.completion",
                )
                logger.debug("InferenceRouter.completion_response", final_response=final_response)
                asyncio.create_task(self.store.store_chat_completion(final_response, messages))
