# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from tests.unit.providers.responses.builtin.fixtures import load_chat_completion_fixture


async def fake_stream(fixture: str = "simple_chat_completion.yaml"):
    value = load_chat_completion_fixture(fixture)
    yield ChatCompletionChunk(
        id=value.id,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    content=c.message.content,
                    role=c.message.role,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id=t.id,
                            function=ChoiceDeltaToolCallFunction(
                                name=t.function.name,
                                arguments=t.function.arguments,
                            ),
                        )
                        for t in (c.message.tool_calls or [])
                    ],
                ),
            )
            for c in value.choices
        ],
        created=1,
        model=value.model,
        object="chat.completion.chunk",
    )
