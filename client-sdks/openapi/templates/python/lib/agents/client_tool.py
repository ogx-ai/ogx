# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import json
from abc import abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import TypedDict

from .types import CompletionMessage, Message, ToolDefinition, ToolResponse


class JSONSchema(TypedDict, total=False):
    type: str
    properties: dict[str, Any]
    required: list[str]


class ClientTool:
    """
    Developers can define their custom tools that models can use
    by extending this class.

    Developers need to provide
        - name
        - description
        - params_definition
        - implement tool's behavior in `run_impl` method

    NOTE: The return of the `run` method needs to be json serializable
    """

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_input_schema(self) -> JSONSchema:
        raise NotImplementedError

    def get_instruction_string(self) -> str:
        return f"Use the function '{self.get_name()}' to: {self.get_description()}"

    def get_tool_definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "name": self.get_name(),
            "description": self.get_description(),
            "parameters": self.get_input_schema(),
        }

    def run(
        self,
        message_history: list[Message],
    ) -> ToolResponse:
        # NOTE: we could override this method to use the entire message history for advanced tools
        last_message = message_history[-1]
        assert isinstance(last_message, CompletionMessage), "Expected CompletionMessage"
        assert len(last_message.tool_calls) == 1, "Expected single tool call"
        tool_call = last_message.tool_calls[0]

        metadata = {}
        try:
            params = json.loads(tool_call.arguments)
            response = self.run_impl(**params)
            if isinstance(response, dict) and "content" in response:
                content = json.dumps(response["content"], ensure_ascii=False)
                metadata = response.get("metadata", {})
            else:
                content = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            content = f"Error when running tool: {e}"
        return ToolResponse(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=content,
            metadata=metadata,
        )

    async def async_run(
        self,
        message_history: list[Message],
    ) -> ToolResponse:
        last_message = message_history[-1]

        assert len(last_message.tool_calls) == 1, "Expected single tool call"
        tool_call = last_message.tool_calls[0]
        metadata = {}
        try:
            params = json.loads(tool_call.arguments)
            response = await self.async_run_impl(**params)
            if isinstance(response, dict) and "content" in response:
                content = json.dumps(response["content"], ensure_ascii=False)
                metadata = response.get("metadata", {})
            else:
                content = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            content = f"Error when running tool: {e}"

        return ToolResponse(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=content,
            metadata=metadata,
        )

    @abstractmethod
    def run_impl(self, **kwargs) -> Any:
        """
        Can return any json serializable object.
        To return metadata along with the response, return a dict with a "content" key, and a "metadata" key, where the "content" is the response that'll
        be serialized and passed to the model, and the "metadata" will be logged as metadata in the tool execution step within the Agent execution trace.
        """
        raise NotImplementedError

    @abstractmethod
    def async_run_impl(self, **kwargs):
        raise NotImplementedError


def _python_type_to_json_schema_type(type_hint: Any) -> str:
    """Convert Python type hints to JSON Schema type strings."""
    # Handle Union types (e.g., Optional[str])
    origin = get_origin(type_hint)
    if origin is Union:
        # Get non-None types from Union
        args = [arg for arg in get_args(type_hint) if arg is not type(None)]
        if args:
            type_hint = args[0]  # Use first non-None type

    # Get the actual type if it's a generic
    if hasattr(type_hint, "__origin__"):
        type_hint = type_hint.__origin__

    # Map Python types to JSON Schema types
    type_name = getattr(type_hint, "__name__", str(type_hint))

    type_mapping = {
        "bool": "boolean",
        "int": "integer",
        "float": "number",
        "str": "string",
        "list": "array",
        "dict": "object",
        "List": "array",
        "Dict": "object",
    }

    return type_mapping.get(type_name, "string")  # Default to string if unknown


def client_tool(func: Callable) -> ClientTool:
    """
    Decorator to convert a function into a ClientTool.
    Usage:
        @client_tool
        def add(x: int, y: int) -> int:
            '''Add 2 integer numbers

            :param x: integer 1
            :param y: integer 2
            :returns: sum of x + y
            '''
            return x + y

    Note that you must use RST-style docstrings with :param tags for each parameter. These will be used for prompting model to use tools correctly.
    :returns: tags in the docstring is optional as it would not be used for the tool's description.

    Your function can return any json serializable object.
    To return metadata along with the response, return a dict with a "content" key, and a "metadata" key, where the "content" is the response that'll
    be serialized and passed to the model, and the "metadata" will be logged as metadata in the tool execution step within the Agent execution trace.
    """

    class _WrappedTool(ClientTool):
        __name__ = func.__name__
        __doc__ = func.__doc__
        __module__ = func.__module__

        def get_name(self) -> str:
            return func.__name__

        def get_description(self) -> str:
            doc = inspect.getdoc(func)
            if doc:
                # Get everything before the first :param
                return doc.split(":param")[0].strip()
            else:
                raise ValueError(
                    f"No description found for client tool {__name__}. Please provide a RST-style docstring with description and :param tags for each parameter."
                )

        def get_input_schema(self) -> JSONSchema:
            hints = get_type_hints(func)
            # Remove return annotation if present
            hints.pop("return", None)

            # Get parameter descriptions from docstring
            properties = {}
            required = []
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""

            for name, type_hint in hints.items():
                # Look for :param name: in docstring
                param_doc = ""
                for line in doc.split("\n"):
                    if line.strip().startswith(f":param {name}:"):
                        param_doc = line.split(":", 2)[2].strip()
                        break

                if param_doc == "":
                    raise ValueError(f"No parameter description found for parameter {name}")

                param = sig.parameters[name]
                is_optional_type = get_origin(type_hint) is Union and type(None) in get_args(type_hint)
                is_required = param.default == inspect.Parameter.empty and not is_optional_type

                properties[name] = {
                    "type": _python_type_to_json_schema_type(type_hint),
                    "description": param_doc,
                }

                if is_required:
                    required.append(name)

            return {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        def run_impl(self, **kwargs) -> Any:
            if inspect.iscoroutinefunction(func):
                raise NotImplementedError("Tool is async but run_impl is not async")
            return func(**kwargs)

        async def async_run_impl(self, **kwargs):
            if inspect.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)

    return _WrappedTool()
