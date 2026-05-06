# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Custom OGX Exception classes should follow the following schema
#   1. All classes should inherit from OGXError (single inheritance only)
#   2. All classes should have a custom error message with the goal of informing the OGX user specifically
#   3. All classes should set a status_code class attribute for HTTP response mapping

import httpx
from pydantic import BaseModel


class OpenAIErrorDetail(BaseModel):
    """Inner error object matching the OpenAI API error format.

    See: https://platform.openai.com/docs/guides/error-codes
    """

    message: str
    type: str | None = None
    code: str | None = None
    param: str | None = None


class OpenAIErrorResponse(BaseModel):
    """Top-level error response matching the OpenAI API error format.

    Usage::

        err = OpenAIErrorResponse.from_message("Not found")
        return JSONResponse(status_code=404, content=err.to_dict())
        await send({"type": "http.response.body", "body": err.to_bytes()})
    """

    error: OpenAIErrorDetail

    @classmethod
    def from_message(
        cls, message: str | Exception, *, type: str | None = None, code: str | None = None
    ) -> "OpenAIErrorResponse":
        """Create an error response from a message string or exception."""
        return cls(error=OpenAIErrorDetail(message=str(message), type=type, code=code))

    def to_dict(self) -> dict:
        """Return a dict suitable for JSONResponse content or SSE events."""
        return self.model_dump(exclude_none=True)

    def to_bytes(self) -> bytes:
        """Return JSON bytes suitable for ASGI send()."""
        return self.model_dump_json(exclude_none=True).encode()


class OGXError(Exception):
    """A base class for all OGX errors with an HTTP status code for API responses."""

    status_code: httpx.codes

    def __init__(self, message: str):
        super().__init__(message)


class ClientListCommand:
    """
    A formatted client list command string.
    Args:
        command: The command to list the resources.
        arguments: The arguments to the command.
        resource_name_plural: The plural name of the resource.

    Returns:
        A formatted client list command string: "Use 'client.files.list()' to list available files."
    """

    def __init__(
        self,
        command: str,
        arguments: list[str] | str | None = None,
        resource_name_plural: str | None = None,
    ):
        self.resource_name_plural = resource_name_plural
        self.command = command
        self.arguments = arguments

    def __str__(self) -> str:
        args_str = ""
        resource_name_str = ""
        if self.arguments:
            if isinstance(self.arguments, list):
                args_str = ", ".join(f'"{arg}"' for arg in self.arguments)
            else:
                args_str = f'"{self.arguments}"'
        if self.resource_name_plural:
            resource_name_str = f" to list available {self.resource_name_plural}"

        return f"Use 'client.{self.command}({args_str})'{resource_name_str}."


class ResourceNotFoundError(OGXError):
    """Raised when a requested OGX resource does not exist.

    Maps to HTTP 404 Not Found. Subclasses (ModelNotFoundError, ResponseNotFoundError,
    etc.) specialize this for specific resource types and often add a client_command
    hint to guide users on how to list available resources.

    :param resource_name: The identifier or name that was not found.
    :param resource_type: Human-readable type label (e.g., "Model", "Response").
    :param client_command: Optional API method to suggest for listing resources (e.g., "files.list").
    :param client_command_args: Optional arguments for the list command. When provided (str or list),
        they are included in the hint, e.g., ``client.connectors.list_tools("conn_123")``. Defaults to
        none (empty parens).
    :param resource_name_plural: Plural label for the "list available X" hint suffix. Defaults to
        ``{resource_type}s`` (e.g., "Models" from "Model"). Override when irregular, e.g., "Batches"
        instead of "Batchs".
    :param parent_resource: Optional context for child resources (e.g., "response 'resp_xyz'").
        When provided, message format is: ``{resource_type} '{resource_name}' not found in {parent_resource}.``
    """

    status_code: httpx.codes = httpx.codes.NOT_FOUND

    def __init__(
        self,
        resource_name: str,
        resource_type: str = "Resource",
        client_command: str | None = None,
        client_command_args: list[str] | str | None = None,
        resource_name_plural: str | None = None,
        parent_resource: str | None = None,
    ) -> None:
        resource_name_plural = resource_name_plural or f"{resource_type}s"

        if parent_resource:
            message = f"{resource_type} '{resource_name}' not found in {parent_resource}."
        else:
            message = f"{resource_type} '{resource_name}' not found."
        if client_command:
            client_list = ClientListCommand(client_command, client_command_args, resource_name_plural)
            message += f" {client_list}"
        super().__init__(message)


class ModelNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced model"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name, resource_type="Model", client_command="models.list")


class VectorStoreNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced vector store"""

    def __init__(self, vector_store_name: str) -> None:
        super().__init__(vector_store_name, resource_type="Vector Store", client_command="vector_dbs.list")


class DatasetNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced dataset"""

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name, resource_type="Dataset", client_command="datasets.list")


class ToolGroupNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced tool group"""

    def __init__(self, toolgroup_name: str) -> None:
        super().__init__(toolgroup_name, resource_type="Tool Group", client_command="toolgroups.list")


class ConversationNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced conversation"""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(conversation_id, resource_type="Conversation")


class ConversationItemNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced item within a conversation"""

    def __init__(self, item_id: str, conversation_id: str) -> None:
        super().__init__(
            item_id,
            resource_type="Conversation item",
            client_command="conversations.items.list",
            client_command_args=conversation_id,
            resource_name_plural="conversation items",
            parent_resource=f"conversation '{conversation_id}'",
        )


class ConnectorNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced connector"""

    def __init__(self, connector_id: str) -> None:
        super().__init__(connector_id, resource_type="Connector", client_command="connectors.list")


class ConnectorToolNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced tool in a connector"""

    def __init__(self, connector_id: str, tool_name: str) -> None:
        super().__init__(
            f"{connector_id}.{tool_name}",
            resource_type="Connector Tool",
            client_command="connectors.list_tools",
            client_command_args=connector_id,
        )


class OpenAIFileObjectNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced file"""

    def __init__(self, file_id: str) -> None:
        super().__init__(file_id, resource_type="File", client_command="files.list")


class BatchNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced batch"""

    def __init__(self, batch_id: str) -> None:
        super().__init__(batch_id, resource_type="Batch", client_command="batches.list", resource_name_plural="batches")


class UnsupportedModelError(OGXError):
    """raised when model is not present in the list of supported models"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, model_name: str, supported_models_list: list[str]):
        message = f"'{model_name}' model is not supported. Supported models are: {', '.join(supported_models_list)}"
        super().__init__(message)


class ModelTypeError(OGXError):
    """raised when a model is present but not the correct type"""

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, model_name: str, model_type: str, expected_model_type: str) -> None:
        message = (
            f"Model '{model_name}' is of type '{model_type}' rather than the expected type '{expected_model_type}'"
        )
        super().__init__(message)


class ConflictError(OGXError):
    """raised when an operation cannot be performed due to a conflict with the current state"""

    status_code: httpx.codes = httpx.codes.CONFLICT

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TokenValidationError(OGXError):
    """raised when token validation fails during authentication"""

    status_code: httpx.codes = httpx.codes.UNAUTHORIZED

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidParameterError(ValueError, OGXError):
    """Raised when a request parameter violates validation constraints.

    Maps to HTTP 400 Bad Request. Use this for client-supplied parameter errors
    such as out-of-range values, invalid formats, or mutually exclusive params.

    :param param_name: Name of the parameter (or comma-separated names for mutually exclusive params).
    :param value: The invalid value that was provided.
    :param constraint: Human-readable description of the constraint (e.g., "Must be >= 1.").
    """

    status_code: httpx.codes = httpx.codes.BAD_REQUEST

    def __init__(self, param_name: str, value: object, constraint: str) -> None:
        message = f"Invalid value for '{param_name}': {value}. {constraint}"
        super().__init__(message)


class ServiceNotEnabledError(OGXError, ValueError):
    """Raised when a required OGX service is not configured or available.

    Maps to HTTP 503 Service Unavailable. Use this when a request depends on a service
    (e.g., Safety API) that has not been enabled in the stack configuration.

    :param service_name: The name of the service that is not enabled (e.g., "Safety API").
    :param provider_specific_message: Optional additional context appended to the message,
        intended for operators or users to understand provider-specific setup steps.
        Separated from the base message by a blank line.
    """

    status_code: httpx.codes = httpx.codes.SERVICE_UNAVAILABLE

    def __init__(self, service_name: str, *, provider_specific_message: str | None = None) -> None:
        message = f"Service '{service_name}' is not enabled. Please check your configuration and enable the service before trying again."
        if provider_specific_message:
            message += f"\n\n{provider_specific_message}"
        super().__init__(message)


class InternalServerError(OGXError):
    """
    A generic server side error that is not caused by the user's request. Sensitive data
    or details of the internal workings of the server should never be exposed to the user.
    Instead, sanitized error information should be logged for debugging purposes.
    """

    status_code: httpx.codes = httpx.codes.INTERNAL_SERVER_ERROR

    def __init__(self, detail: str | None = None) -> None:
        message = detail or "An internal error occurred while processing your request."
        super().__init__(message)


class ResponseNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced response"""

    def __init__(self, response_id: str) -> None:
        super().__init__(response_id, resource_type="Response", client_command="responses.list")


class ResponseInputItemNotFoundError(ResourceNotFoundError):
    """raised when OGX cannot find a referenced input item within a response"""

    def __init__(self, item_id: str, response_id: str) -> None:
        super().__init__(
            item_id,
            resource_type="Input item",
            client_command="responses.input_items.list",
            client_command_args=response_id,
            resource_name_plural="input items",
            parent_resource=f"response '{response_id}'",
        )


class FileTooLargeError(OGXError):
    """raised when an uploaded file exceeds the maximum allowed size"""

    status_code: httpx.codes = httpx.codes.REQUEST_ENTITY_TOO_LARGE

    def __init__(self, file_size: int, max_size: int) -> None:
        message = (
            f"File size {file_size} bytes exceeds the maximum allowed upload size of {max_size} bytes "
            f"({max_size / (1024 * 1024):.0f} MB)"
        )
        super().__init__(message)
