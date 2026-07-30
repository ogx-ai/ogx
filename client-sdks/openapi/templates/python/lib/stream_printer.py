# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .agents.event_logger import TurnStreamEventPrinter
from .inference.event_logger import InferenceStreamLogEventPrinter


class EventStreamPrinter:
    @classmethod
    def gen(cls, event_generator):
        inference_printer = None
        turn_printer = None
        for chunk in event_generator:
            if not hasattr(chunk, "event"):
                raise ValueError(f"Unexpected chunk without event: {chunk}")

            event = chunk.event
            if hasattr(event, "event_type"):
                if not inference_printer:
                    inference_printer = InferenceStreamLogEventPrinter()
                yield from inference_printer.yield_printable_events(chunk)
            elif hasattr(event, "payload") and hasattr(event.payload, "event_type"):
                if not turn_printer:
                    turn_printer = TurnStreamEventPrinter()
                yield from turn_printer.yield_printable_events(chunk)
            else:
                raise ValueError(f"Unsupported event: {event}")
