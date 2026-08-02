# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ogx_api.skills import Skills

from opentelemetry import metrics

from ogx.core.datatypes import AccessRule
from ogx.log import get_logger
from ogx.providers.utils.responses.responses_store import ResponsesStore
from ogx.telemetry.constants import (
    RESPONSES_AGENTIC_CALLS_TOTAL,
    RESPONSES_PARAMETER_USAGE_TOTAL,
    RESPONSES_TOOL_TYPES_USED_TOTAL,
)
from ogx_api import (
    CancelResponseRequest,
    CompactResponseRequest,
    Connectors,
    Conversations,
    CreateResponseRequest,
    DeleteResponseRequest,
    Files,
    Inference,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    OpenAICompactedResponse,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    Prompts,
    Responses,
    RetrieveResponseRequest,
    ToolGroups,
    ToolRuntime,
    VectorIO,
    WebSearchToolTypes,
)

from .config import BuiltinResponsesImplConfig
from .responses.openai_responses import OpenAIResponsesImpl

logger = get_logger(name=__name__, category="agents::builtin")

_meter = metrics.get_meter("ogx.responses", version="1.0.0")

_parameter_usage_total = _meter.create_counter(
    name=RESPONSES_PARAMETER_USAGE_TOTAL,
    description="Tracks which optional parameters are explicitly provided in Responses API calls",
    unit="1",
)

_tool_types_used_total = _meter.create_counter(
    name=RESPONSES_TOOL_TYPES_USED_TOTAL,
    description="Counts tool types present in Responses API calls",
    unit="1",
)

_agentic_calls_total = _meter.create_counter(
    name=RESPONSES_AGENTIC_CALLS_TOTAL,
    description="Total Responses API calls that include tools (agentic calls)",
    unit="1",
)

_REQUIRED_FIELDS = {"input", "model"}


def _record_parameter_usage(request: CreateResponseRequest, operation: str) -> None:
    """Record which optional parameters were explicitly provided in the request."""
    declared_fields = set(request.model_fields.keys())
    for field_name in (request.model_fields_set & declared_fields) - _REQUIRED_FIELDS:
        _parameter_usage_total.add(1, {"operation": operation, "parameter": field_name})


def _record_tool_usage(request: CreateResponseRequest) -> None:
    """Record tool type usage and agentic call counts."""
    if not request.tools:
        return
    _agentic_calls_total.add(1)
    seen_types: set[str] = set()
    for tool in request.tools:
        tool_type = tool.type
        if tool_type in WebSearchToolTypes:
            tool_type = "web_search"
        if tool_type not in seen_types:
            seen_types.add(tool_type)
            _tool_types_used_total.add(1, {"tool_type": tool_type})


class BuiltinResponsesImpl(Responses):
    """Built-in responses implementing the Agents API with tool use and responses support."""

    def __init__(
        self,
        config: BuiltinResponsesImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        conversations_api: Conversations,
        prompts_api: Prompts,
        files_api: Files,
        connectors_api: Connectors,
        policy: list[AccessRule],
        skills_api: "Skills | None" = None,
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api
        self.conversations_api = conversations_api
        self.prompts_api = prompts_api
        self.files_api = files_api
        self.openai_responses_impl: OpenAIResponsesImpl | None = None
        self.policy = policy
        self.connectors_api = connectors_api
        self.skills_api = skills_api

    async def initialize(self) -> None:
        self.responses_store = ResponsesStore(self.config.persistence.responses, self.policy)
        await self.responses_store.initialize()
        if not self.responses_store.sql_store:
            raise RuntimeError("Responses store is not initialized")
        self.openai_responses_impl = OpenAIResponsesImpl(
            inference_api=self.inference_api,
            tool_groups_api=self.tool_groups_api,
            tool_runtime_api=self.tool_runtime_api,
            responses_store=self.responses_store,
            vector_io_api=self.vector_io_api,
            moderation_endpoint=self.config.moderation_endpoint,
            moderation_headers=self.config.moderation_headers,
            conversations_api=self.conversations_api,
            prompts_api=self.prompts_api,
            files_api=self.files_api,
            vector_stores_config=self.config.vector_stores_config,
            connectors_api=self.connectors_api,
            skills_api=self.skills_api,
            compaction_config=self.config.compaction_config,
            memory_config=self.config.memory_config,
        )
        await self.openai_responses_impl.initialize()

    async def shutdown(self) -> None:
        if self.openai_responses_impl is not None:
            await self.openai_responses_impl.shutdown()

    # OpenAI responses
    async def get_openai_response(
        self,
        request: RetrieveResponseRequest,
    ) -> OpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.get_openai_response(request.response_id)

    async def create_openai_response(
        self,
        request: CreateResponseRequest,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        """Create an OpenAI response.

        Returns either a single response object (non-streaming) or an async iterator
        yielding response stream events (streaming).
        """
        _record_parameter_usage(request, operation="create_response")
        _record_tool_usage(request)
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        result = await self.openai_responses_impl.create_openai_response(request)
        return result

    async def list_openai_responses(
        self,
        request: ListResponsesRequest,
    ) -> ListOpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.list_openai_responses(
            request.after, request.limit, request.model, request.order
        )

    async def list_openai_response_input_items(
        self,
        request: ListResponseInputItemsRequest,
    ) -> ListOpenAIResponseInputItem:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.list_openai_response_input_items(
            request.response_id,
            request.after,
            request.before,
            request.include,
            request.limit,
            request.order,
        )

    async def compact_openai_response(
        self,
        request: CompactResponseRequest,
    ) -> OpenAICompactedResponse:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.compact_openai_response(
            model=request.model,
            input=request.input,
            instructions=request.instructions,
            previous_response_id=request.previous_response_id,
            prompt_cache_key=request.prompt_cache_key,
            extra_body=request.model_extra,
        )

    async def delete_openai_response(
        self,
        request: DeleteResponseRequest,
    ) -> OpenAIDeleteResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.delete_openai_response(request.response_id)

    async def cancel_openai_response(
        self,
        request: CancelResponseRequest,
    ) -> OpenAIResponseObject:
        assert self.openai_responses_impl is not None, "OpenAI responses not initialized"
        return await self.openai_responses_impl.cancel_openai_response(request.response_id)
