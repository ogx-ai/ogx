# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
This file contains constants used for naming data captured for telemetry.

This is used to ensure that the data captured for telemetry is consistent and can be used to
identify and correlate data. If custom telemetry data is added to ogx, please add
constants for it here.
"""

ogx_prefix = "ogx"

# Safety Attributes
RUN_SHIELD_OPERATION_NAME = "run_shield"

SAFETY_REQUEST_PREFIX = f"{ogx_prefix}.safety.request"
SAFETY_REQUEST_SHIELD_ID_ATTRIBUTE = f"{SAFETY_REQUEST_PREFIX}.shield_id"
SAFETY_REQUEST_MESSAGES_ATTRIBUTE = f"{SAFETY_REQUEST_PREFIX}.messages"

SAFETY_RESPONSE_PREFIX = f"{ogx_prefix}.safety.response"
SAFETY_RESPONSE_METADATA_ATTRIBUTE = f"{SAFETY_RESPONSE_PREFIX}.metadata"
SAFETY_RESPONSE_VIOLATION_LEVEL_ATTRIBUTE = f"{SAFETY_RESPONSE_PREFIX}.violation.level"
SAFETY_RESPONSE_USER_MESSAGE_ATTRIBUTE = f"{SAFETY_RESPONSE_PREFIX}.violation.user_message"

# Tool Runtime Metrics
# These constants define the names for OpenTelemetry metrics tracking tool runtime operations
TOOL_RUNTIME_PREFIX = f"{ogx_prefix}.tool_runtime"

# Tool invocation metrics
TOOL_INVOCATIONS_TOTAL = f"{TOOL_RUNTIME_PREFIX}.invocations_total"
TOOL_DURATION = f"{TOOL_RUNTIME_PREFIX}.duration_seconds"

# Vector IO Metrics
# These constants define the names for OpenTelemetry metrics tracking vector store operations
VECTOR_IO_PREFIX = f"{ogx_prefix}.vector_io"

# Vector operation counters
VECTOR_INSERTS_TOTAL = f"{VECTOR_IO_PREFIX}.inserts_total"
VECTOR_QUERIES_TOTAL = f"{VECTOR_IO_PREFIX}.queries_total"
VECTOR_DELETES_TOTAL = f"{VECTOR_IO_PREFIX}.deletes_total"
VECTOR_STORES_TOTAL = f"{VECTOR_IO_PREFIX}.stores_total"
VECTOR_FILES_TOTAL = f"{VECTOR_IO_PREFIX}.files_total"
VECTOR_CHUNKS_PROCESSED_TOTAL = f"{VECTOR_IO_PREFIX}.chunks_processed_total"

# Vector operation durations
VECTOR_INSERT_DURATION = f"{VECTOR_IO_PREFIX}.insert_duration_seconds"
VECTOR_RETRIEVAL_DURATION = f"{VECTOR_IO_PREFIX}.retrieval_duration_seconds"

# Request Metrics
# These constants define the names for OpenTelemetry metrics tracking API gateway request-level operations
REQUEST_PREFIX = f"{ogx_prefix}.request"

REQUESTS_TOTAL = f"{REQUEST_PREFIX}s_total"
REQUEST_DURATION_SECONDS = f"{REQUEST_PREFIX}_duration_seconds"
CONCURRENT_REQUESTS = f"{ogx_prefix}.concurrent_requests"

# Inference Metrics
# These constants define the names for OpenTelemetry metrics tracking inference operations
INFERENCE_PREFIX = f"{ogx_prefix}.inference"

INFERENCE_DURATION = f"{INFERENCE_PREFIX}.duration_seconds"
INFERENCE_TIME_TO_FIRST_TOKEN = f"{INFERENCE_PREFIX}.time_to_first_token_seconds"
INFERENCE_TOKENS_PER_SECOND = f"{INFERENCE_PREFIX}.tokens_per_second"

# Responses API Metrics
RESPONSES_PREFIX = f"{ogx_prefix}.responses"
RESPONSES_PARAMETER_USAGE_TOTAL = f"{RESPONSES_PREFIX}.parameter_usage_total"
