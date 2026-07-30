# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import abstractmethod

from .types import CompletionMessage, ToolCall


class ToolParser:
    """
    Abstract base class for parsing agent responses into tool calls. Implement this class to customize how
    agent outputs are processed and transformed into executable tool calls.

    To use this class:
    1. Create a subclass of ToolParser
    2. Implement the `get_tool_calls` method
    3. Pass your parser instance to the Agent's constructor

    Example:
        class MyCustomParser(ToolParser):
            def get_tool_calls(self, output_message: CompletionMessage) -> list[ToolCall]:
                # Add your custom parsing logic here
                return extracted_tool_calls

    Methods:
        get_tool_calls(output_message: CompletionMessage) -> list[ToolCall]:
            Abstract method that must be implemented by subclasses to process
            the agent's response and extract tool calls.

            Args:
                output_message (CompletionMessage): The response message from agent turn

            Returns:
                list[ToolCall] | None: A list of parsed tool calls, or None if no tools should be called
    """

    @abstractmethod
    def get_tool_calls(self, output_message: CompletionMessage) -> list[ToolCall]:
        raise NotImplementedError
