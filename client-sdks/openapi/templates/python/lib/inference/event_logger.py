# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import Generator

from termcolor import cprint

from ...models.open_ai_chat_completion_chunk import OpenAIChatCompletionChunk as ChatCompletionChunk


class InferenceStreamPrintableEvent:
    def __init__(
        self,
        content: str = "",
        end: str = "\n",
        color="white",
    ):
        self.content = content
        self.color = color
        self.end = "\n" if end is None else end

    def print(self, flush=True):
        cprint(f"{self.content}", color=self.color, end=self.end, flush=flush)


class InferenceStreamLogEventPrinter:
    def __init__(self):
        self.is_thinking = False

    def yield_printable_events(
        self, chunk: ChatCompletionChunk
    ) -> Generator[InferenceStreamPrintableEvent, None, None]:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            yield from self._handle_chat_completion_chunk(chunk)

    def _handle_chat_completion_chunk(
        self, chunk: ChatCompletionChunk
    ) -> Generator[InferenceStreamPrintableEvent, None, None]:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta:
            if delta.role:
                yield InferenceStreamPrintableEvent(f"{delta.role}> ", color="cyan", end="")
            if delta.content:
                yield InferenceStreamPrintableEvent(delta.content, color="yellow", end="")
            if choice.finish_reason:
                if choice.finish_reason == "length":
                    yield InferenceStreamPrintableEvent("<truncated>", color="red", end="")
                yield InferenceStreamPrintableEvent()


class EventLogger:
    def log(self, event_generator):
        printer = InferenceStreamLogEventPrinter()
        for chunk in event_generator:
            yield from printer.yield_printable_events(chunk)
