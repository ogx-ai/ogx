# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from opentelemetry import trace

from ogx_api import OpenAIMessageParam, RunShieldResponse

from .constants import (
    RUN_SHIELD_OPERATION_NAME,
    SAFETY_REQUEST_MESSAGES_ATTRIBUTE,
    SAFETY_REQUEST_SHIELD_ID_ATTRIBUTE,
    SAFETY_RESPONSE_METADATA_ATTRIBUTE,
    SAFETY_RESPONSE_USER_MESSAGE_ATTRIBUTE,
    SAFETY_RESPONSE_VIOLATION_LEVEL_ATTRIBUTE,
)


def safety_span_name(shield_id: str) -> str:
    """Build the OpenTelemetry span name for a safety shield invocation.

    Args:
        shield_id: Identifier of the shield being invoked.

    Returns:
        The formatted span name combining the operation name and shield id.
    """
    return f"{RUN_SHIELD_OPERATION_NAME} {shield_id}"


# TODO: Consider using Wrapt to automatically instrument code
# This is the industry standard way to package automatically instrumentation in python.
def safety_request_span_attributes(
    shield_id: str, messages: list[OpenAIMessageParam], response: RunShieldResponse
) -> None:
    """Set OpenTelemetry span attributes for a safety shield request and response.

    Args:
        shield_id: Identifier of the shield that was invoked.
        messages: The list of messages that were checked by the shield.
        response: The shield's response, potentially containing violation details.
    """
    span = trace.get_current_span()
    span.set_attribute(SAFETY_REQUEST_SHIELD_ID_ATTRIBUTE, shield_id)
    messages_json = json.dumps([msg.model_dump() for msg in messages])
    span.set_attribute(SAFETY_REQUEST_MESSAGES_ATTRIBUTE, messages_json)

    if response.violation:
        if response.violation.metadata:
            metadata_json = json.dumps(response.violation.metadata)
            span.set_attribute(SAFETY_RESPONSE_METADATA_ATTRIBUTE, metadata_json)
        if response.violation.user_message:
            span.set_attribute(SAFETY_RESPONSE_USER_MESSAGE_ATTRIBUTE, response.violation.user_message)
        span.set_attribute(SAFETY_RESPONSE_VIOLATION_LEVEL_ATTRIBUTE, response.violation.violation_level.value)
