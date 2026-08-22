#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAI Responses Schema Drift Checker

Compares Pydantic models in ogx_api.openai_responses against the vendored
OpenAI spec (docs/static/openai-spec-*.yml) to detect unintentional drift.

Reports:
- Fields present in spec but missing from Pydantic models
- Fields present in Pydantic models but missing from spec
- New spec schemas with no corresponding Pydantic model
- Streaming event types in spec not covered by the stream union

Intentional divergences are documented inline and suppressed from output.

Usage:
    uv run python scripts/check_openai_responses_drift.py
    uv run python scripts/check_openai_responses_drift.py --spec docs/static/openai-spec-2.3.0.yml
    uv run python scripts/check_openai_responses_drift.py --strict  # exit 1 on any drift
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT / "docs" / "static"
_SPEC_FILENAME_PATTERN = re.compile(r"^openai-spec-(\d+(?:\.\d+)*)\.yml$")

# ---------------------------------------------------------------------------
# Base mapping: OpenAI spec schema name -> OGX Pydantic class name.
# Stream event mappings are derived automatically from ResponseStreamEvent.
# ---------------------------------------------------------------------------
BASE_SCHEMA_TO_PYDANTIC: dict[str, str] = {
    # Core response
    "ResponseError": "OpenAIResponseError",
    "ResponseUsage": "OpenAIResponseUsage",
    # Input content types
    "InputTextContent": "OpenAIResponseInputMessageContentText",
    "InputImageContent": "OpenAIResponseInputMessageContentImage",
    "InputFileContent": "OpenAIResponseInputMessageContentFile",
    # Output content types
    "OutputTextContent": "OpenAIResponseOutputMessageContentOutputText",
    "RefusalContent": "OpenAIResponseContentPartRefusal",
    # Messages
    "OutputMessage": "OpenAIResponseMessage",
    "InputMessage": "OpenAIResponseMessage",
    # Tool calls (output)
    "WebSearchToolCall": "OpenAIResponseOutputMessageWebSearchToolCall",
    "FileSearchToolCall": "OpenAIResponseOutputMessageFileSearchToolCall",
    "FunctionToolCall": "OpenAIResponseOutputMessageFunctionToolCall",
    "FunctionToolCallOutput": "OpenAIResponseInputFunctionToolCallOutput",
    "MCPToolCall": "OpenAIResponseOutputMessageMCPCall",
    "MCPListTools": "OpenAIResponseOutputMessageMCPListTools",
    "MCPListToolsTool": "MCPListToolsTool",
    "MCPApprovalRequest": "OpenAIResponseMCPApprovalRequest",
    "MCPApprovalResponse": "OpenAIResponseMCPApprovalResponse",
    # Reasoning
    "ReasoningItem": "OpenAIResponseOutputMessageReasoningItem",
    # Tool definitions (input)
    "WebSearchTool": "OpenAIResponseInputToolWebSearch",
    "FunctionTool": "OpenAIResponseInputToolFunction",
    "FileSearchTool": "OpenAIResponseInputToolFileSearch",
    "MCPTool": "OpenAIResponseInputToolMCP",
}

# Stream event names whose model class cannot be inferred with the standard
# "OpenAIResponseObjectStream{SpecNameWithoutEvent}" convention.
_STREAM_EVENT_MODEL_OVERRIDES: dict[str, str] = {
    "ResponseErrorEvent": "OpenAIResponseObjectStreamError",
    "ResponseTextDeltaEvent": "OpenAIResponseObjectStreamResponseOutputTextDelta",
    "ResponseTextDoneEvent": "OpenAIResponseObjectStreamResponseOutputTextDone",
}

# ---------------------------------------------------------------------------
# Spec schemas that OGX intentionally does not implement.
# Each entry documents the reason.
# ---------------------------------------------------------------------------
INTENTIONALLY_SKIPPED: dict[str, str] = {
    # Desktop/agent-specific tool types not supported by OGX
    "ComputerToolCall": "Desktop automation tool, not supported",
    "ComputerToolCallOutput": "Desktop automation tool output, not supported",
    "ComputerToolCallOutputResource": "Desktop automation tool output resource, not supported",
    "LocalShellToolCall": "Local shell tool, not supported",
    "LocalShellToolCallOutput": "Local shell tool output, not supported",
    "ApplyPatchToolCall": "Apply patch tool, not supported",
    "ApplyPatchToolCallOutput": "Apply patch tool output, not supported",
    "FunctionShellCall": "Function shell call, not supported",
    "FunctionShellCallOutput": "Function shell call output, not supported",
    "ImageGenToolCall": "Image generation tool, not supported",
    "CodeInterpreterToolCall": "Code interpreter tool, not supported",
    "ToolSearchCall": "Tool search call, not supported",
    "ToolSearchOutput": "Tool search output, not supported",
    "CustomToolCall": "Custom tool call, not supported",
    "CustomToolCallOutputResource": "Custom tool call output resource, not supported",
    # Audio streaming events not supported by OGX
    "ResponseAudioDeltaEvent": "Audio streaming not supported",
    "ResponseAudioDoneEvent": "Audio streaming not supported",
    "ResponseAudioTranscriptDeltaEvent": "Audio streaming not supported",
    "ResponseAudioTranscriptDoneEvent": "Audio streaming not supported",
    # Code interpreter streaming events
    "ResponseCodeInterpreterCallCodeDeltaEvent": "Code interpreter not supported",
    "ResponseCodeInterpreterCallCodeDoneEvent": "Code interpreter not supported",
    "ResponseCodeInterpreterCallCompletedEvent": "Code interpreter not supported",
    "ResponseCodeInterpreterCallInProgressEvent": "Code interpreter not supported",
    "ResponseCodeInterpreterCallInterpretingEvent": "Code interpreter not supported",
    # Image generation streaming events
    "ResponseImageGenCallCompletedEvent": "Image generation not supported",
    "ResponseImageGenCallGeneratingEvent": "Image generation not supported",
    "ResponseImageGenCallInProgressEvent": "Image generation not supported",
    "ResponseImageGenCallPartialImageEvent": "Image generation not supported",
    # Custom tool call streaming events
    "ResponseCustomToolCallInputDeltaEvent": "Custom tool call not supported",
    "ResponseCustomToolCallInputDoneEvent": "Custom tool call not supported",
    # Queued event
    "ResponseQueuedEvent": "Background/queued responses not supported",
    # Meta schemas (not direct models)
    "ResponseStreamEvent": "Union type, checked separately via streaming event coverage",
    "ResponseStreamOptions": "Chat completions stream options, not Responses API",
    "ResponseProperties": "Partial schema merged into Response via allOf",
    "ModelResponseProperties": "Partial schema merged into Response via allOf",
    "Response": "Assembled from allOf, checked as OpenAIResponseObject",
    "ResponseModalities": "Enum, inlined into request handling",
    "ResponsePromptVariables": "Prompt variable mapping, handled inline",
    "ResponseTextParam": "Maps to OpenAIResponseText but via TypedDict",
    "ResponseFormatJsonObject": "Response format types, handled in inference models",
    "ResponseFormatJsonSchema": "Response format types, handled in inference models",
    "ResponseFormatJsonSchemaSchema": "Response format types, handled in inference models",
    "ResponseFormatText": "Response format types, handled in inference models",
    "ResponseFormatTextGrammar": "Response format types, handled in inference models",
    "ResponseFormatTextPython": "Response format types, handled in inference models",
    "ResponseItemList": "Pagination wrapper, handled as ListOpenAIResponseInputItem",
    "ResponseLogProb": "Log probability type, imported from ogx_api.inference",
    "ResponseOutputText": "Output text helper, handled as OpenAIResponseOutputMessageContentOutputText",
    # Enums inlined into field types
    "ResponseErrorCode": "Enum type, inlined into ResponseError.code",
    # MCP schemas handled differently
    "MCPApprovalResponseResource": "Output variant of MCPApprovalResponse, mapped differently",
    "MCPToolFilter": "MCP tool filter, handled as AllowedToolsFilter",
    "MCPToolCallStatus": "Enum, inlined",
    # Realtime and webhook schemas are separate APIs
    "CompactionBody": "Handled as OpenAIResponseCompaction with different field names",
    "CompactResponseMethodPublicBody": "Request body schema, not a response model",
    # Client/server event wrappers
    "ResponsesClientEvent": "WebSocket client event wrapper, not implemented",
    "ResponsesClientEventResponseCreate": "WebSocket client event, not implemented",
    "ResponsesServerEvent": "WebSocket server event wrapper, not implemented",
}

# ---------------------------------------------------------------------------
# Known intentional field-level divergences.
# Key: (spec_schema_name, field_name) -> reason
# ---------------------------------------------------------------------------
INTENTIONAL_FIELD_DIVERGENCES: dict[tuple[str, str], str] = {
    # OGX uses Pydantic model_validator instead of spec-level validation
    ("InputFileContent", "file_data"): "OGX adds file_data field for inline file content",
    ("InputFileContent", "file_url"): "OGX adds file_url field for URL-based file content",
    # OGX combines InputMessage + InputMessageResource into one model
    ("InputMessage", "id"): "OGX merges InputMessage and InputMessageResource into one model",
    # WebSearchToolCall has extra fields in spec that OGX omits
    ("WebSearchToolCall", "action"): "OGX does not surface search action details",
    # FunctionToolCall has namespace in spec, OGX omits it
    ("FunctionToolCall", "namespace"): "OGX does not support function namespaces",
    # OutputMessage has phase field in spec, OGX omits it
    ("OutputMessage", "phase"): "OGX does not support message phases",
    # OpenAI MCP tool has additional fields OGX handles differently
    ("MCPTool", "server_url"): "OGX uses server_url on input tool, not output tool",
    ("MCPTool", "headers"): "OGX handles headers on input tool",
    ("MCPTool", "require_approval"): "OGX handles require_approval on input tool",
    ("MCPTool", "connector_id"): "OGX handles connector_id on input tool only",
    ("MCPTool", "authorization"): "OGX handles authorization on input tool only",
    # MCPApprovalResponse spec has request_id, OGX uses approval_request_id
    ("MCPApprovalResponse", "request_id"): "OGX uses approval_request_id instead",
    # OGX adds response_id to some streaming events for convenience
    ("ResponseOutputItemAddedEvent", "response_id"): "OGX adds response_id for event correlation",
    ("ResponseOutputItemDoneEvent", "response_id"): "OGX adds response_id for event correlation",
    ("ResponseContentPartAddedEvent", "response_id"): "OGX adds response_id for event correlation",
    ("ResponseContentPartDoneEvent", "response_id"): "OGX adds response_id for event correlation",
}


def _parse_spec_version(path: Path) -> tuple[int, ...] | None:
    match = _SPEC_FILENAME_PATTERN.match(path.name)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _spec_sort_key(path: Path) -> tuple[bool, tuple[int, ...], str]:
    version = _parse_spec_version(path)
    # Prefer files with a parseable version, then the highest semantic version.
    return (version is not None, version or (-1,), path.name)


def _find_spec_file(spec_path: str | None) -> Path:
    if spec_path:
        return Path(spec_path)
    candidates = list(SPEC_DIR.glob("openai-spec-*.yml"))
    if not candidates:
        print("ERROR: No openai-spec-*.yml found in docs/static/", file=sys.stderr)
        sys.exit(1)
    return max(candidates, key=_spec_sort_key)


def _load_spec(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _get_schema_properties(schema: dict[str, Any], all_schemas: dict[str, Any]) -> set[str]:
    """Extract property names from a spec schema, resolving $ref, allOf, and anyOf-with-null."""
    props: set[str] = set()

    if "properties" in schema:
        props.update(schema["properties"].keys())

    if "allOf" in schema:
        for sub in schema["allOf"]:
            if "$ref" in sub:
                ref_name = sub["$ref"].split("/")[-1]
                if ref_name in all_schemas:
                    props.update(_get_schema_properties(all_schemas[ref_name], all_schemas))
            else:
                props.update(_get_schema_properties(sub, all_schemas))

    # Handle anyOf-with-null pattern: anyOf: [{type: object, properties: ...}, {type: null}]
    if "anyOf" in schema and not props:
        for sub in schema["anyOf"]:
            if isinstance(sub, dict) and sub.get("type") == "object" and "properties" in sub:
                props.update(_get_schema_properties(sub, all_schemas))

    return props


@cache
def _get_openai_responses_module() -> Any:
    import ogx_api.openai_responses as mod

    return mod


def _get_pydantic_model(cls_name: str) -> Any | None:
    mod = _get_openai_responses_module()
    return getattr(mod, cls_name, None)


def _get_pydantic_fields(cls_name: str) -> set[str]:
    """Get field names from a Pydantic model by importing and inspecting it."""
    cls = _get_pydantic_model(cls_name)
    if cls is None:
        return set()

    if hasattr(cls, "model_fields"):
        return set(cls.model_fields.keys())
    if hasattr(cls, "__annotations__"):
        return set(cls.__annotations__.keys())
    return set()


def _check_field_drift(
    spec_name: str,
    pydantic_name: str,
    spec_props: set[str],
    pydantic_fields: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Compare fields between spec and Pydantic model. Return (missing, extra) as structured dicts."""
    missing = []
    extra = []

    for field in sorted(spec_props - pydantic_fields):
        if (spec_name, field) in INTENTIONAL_FIELD_DIVERGENCES:
            continue
        missing.append({"spec_schema": spec_name, "field": field, "pydantic_model": pydantic_name})

    for field in sorted(pydantic_fields - spec_props):
        if (spec_name, field) in INTENTIONAL_FIELD_DIVERGENCES:
            continue
        extra.append({"spec_schema": spec_name, "field": field, "pydantic_model": pydantic_name})

    return missing, extra


def _get_stream_event_members(all_schemas: dict[str, Any]) -> list[str]:
    """Return all schema names referenced by ResponseStreamEvent."""
    stream_event = all_schemas.get("ResponseStreamEvent", {})
    members: list[str] = []
    for key in ("anyOf", "oneOf"):
        for ref in stream_event.get(key, []):
            if "$ref" in ref:
                members.append(ref["$ref"].split("/")[-1])
    return members


def _resolve_stream_event_model_name(spec_schema_name: str) -> str | None:
    """Infer the stream event model class name for a spec schema name."""
    override = _STREAM_EVENT_MODEL_OVERRIDES.get(spec_schema_name)
    if override is not None:
        return override
    if not spec_schema_name.startswith("Response") or not spec_schema_name.endswith("Event"):
        return None

    stem = spec_schema_name.removesuffix("Event")
    candidates = (
        f"OpenAIResponseObjectStream{stem}",
        f"OpenAIResponseObjectStream{stem}".replace("MCP", "Mcp"),
    )
    for candidate in candidates:
        if _get_pydantic_model(candidate) is not None:
            return candidate
    return None


def _build_schema_mapping(all_schemas: dict[str, Any]) -> dict[str, str]:
    """Build schema->model mapping from a small manual list plus stream-event inference."""
    schema_to_pydantic = dict(BASE_SCHEMA_TO_PYDANTIC)
    for stream_member in _get_stream_event_members(all_schemas):
        if stream_member in schema_to_pydantic or stream_member in INTENTIONALLY_SKIPPED:
            continue
        model_name = _resolve_stream_event_model_name(stream_member)
        if model_name is not None:
            schema_to_pydantic[stream_member] = model_name
    return schema_to_pydantic


def _check_streaming_coverage(all_schemas: dict[str, Any], schema_to_pydantic: dict[str, str]) -> list[dict[str, str]]:
    """Check that all ResponseStreamEvent members are covered."""
    uncovered = []
    for member in _get_stream_event_members(all_schemas):
        if member not in schema_to_pydantic and member not in INTENTIONALLY_SKIPPED:
            uncovered.append({"spec_schema": member})

    return uncovered


# All non-endpoint Response-related schema names to check in the spec
_EXTRA_SCHEMA_NAMES = (
    "MCPApprovalRequest",
    "MCPApprovalResponse",
    "MCPApprovalResponseResource",
    "MCPListTools",
    "MCPListToolsTool",
    "MCPTool",
    "MCPToolCall",
    "MCPToolFilter",
    "MCPToolCallStatus",
    "OutputMessage",
    "InputMessage",
    "OutputTextContent",
    "RefusalContent",
    "InputTextContent",
    "InputImageContent",
    "InputFileContent",
    "WebSearchToolCall",
    "FileSearchToolCall",
    "FunctionToolCall",
    "ReasoningItem",
    "CompactionBody",
    "CompactResponseMethodPublicBody",
    "WebSearchTool",
    "FunctionTool",
    "FileSearchTool",
    "CustomToolCall",
    "CustomToolCallOutputResource",
    "ComputerToolCall",
    "ComputerToolCallOutput",
    "ComputerToolCallOutputResource",
    "LocalShellToolCall",
    "LocalShellToolCallOutput",
    "ApplyPatchToolCall",
    "ApplyPatchToolCallOutput",
    "FunctionShellCall",
    "FunctionShellCallOutput",
    "ImageGenToolCall",
    "CodeInterpreterToolCall",
    "ToolSearchCall",
    "ToolSearchOutput",
    "FunctionToolCallOutput",
)


def _run_checks(spec_path: Path) -> dict[str, Any]:
    """Run all drift checks and return structured results."""
    spec = _load_spec(spec_path)
    all_schemas = spec.get("components", {}).get("schemas", {})
    schema_to_pydantic = _build_schema_mapping(all_schemas)

    response_schemas = {
        name: schema
        for name, schema in all_schemas.items()
        if name.startswith("Response") or name in _EXTRA_SCHEMA_NAMES
    }

    unmapped: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    missing_fields: list[dict[str, str]] = []
    extra_fields: list[dict[str, str]] = []
    stale_mappings: list[dict[str, str]] = []
    stream_gaps: list[dict[str, str]] = []

    for name in sorted(response_schemas):
        if name in schema_to_pydantic:
            continue
        if name in INTENTIONALLY_SKIPPED:
            skipped.append({"spec_schema": name, "reason": INTENTIONALLY_SKIPPED[name]})
            continue
        unmapped.append({"spec_schema": name})

    for spec_name, pydantic_name in sorted(schema_to_pydantic.items()):
        if spec_name not in all_schemas:
            stale_mappings.append({"spec_schema": spec_name, "pydantic_model": pydantic_name})
            continue

        spec_props = _get_schema_properties(all_schemas[spec_name], all_schemas)
        pydantic_fields = _get_pydantic_fields(pydantic_name)

        if not pydantic_fields:
            stale_mappings.append({"spec_schema": spec_name, "pydantic_model": pydantic_name, "error": "import_failed"})
            continue

        missing, extra = _check_field_drift(spec_name, pydantic_name, spec_props, pydantic_fields)
        missing_fields.extend(missing)
        extra_fields.extend(extra)

    stream_gaps = _check_streaming_coverage(all_schemas, schema_to_pydantic)

    intentional_divergences = [
        {"spec_schema": spec_name, "field": field, "reason": reason}
        for (spec_name, field), reason in sorted(INTENTIONAL_FIELD_DIVERGENCES.items())
    ]

    total_issues = len(unmapped) + len(missing_fields) + len(extra_fields) + len(stale_mappings) + len(stream_gaps)

    return {
        "spec_file": spec_path.name,
        "summary": {
            "mapped_schemas": len(schema_to_pydantic),
            "skipped_schemas": len(INTENTIONALLY_SKIPPED),
            "response_schemas_in_spec": len(response_schemas),
            "total_issues": total_issues,
            "missing_fields": len(missing_fields),
            "extra_fields": len(extra_fields),
            "unmapped_schemas": len(unmapped),
            "stale_mappings": len(stale_mappings),
            "streaming_event_gaps": len(stream_gaps),
        },
        "unmapped_schemas": unmapped,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "stale_mappings": stale_mappings,
        "streaming_event_gaps": stream_gaps,
        "intentionally_skipped": skipped,
        "intentional_field_divergences": intentional_divergences,
    }


def _print_report(results: dict[str, Any], *, verbose: bool = False) -> None:
    """Print human-readable report to stdout."""
    s = results["summary"]
    print(f"Spec: {results['spec_file']}")
    print(f"Mapped schemas: {s['mapped_schemas']}")
    print(f"Skipped schemas: {s['skipped_schemas']}")
    print(f"Response-related schemas in spec: {s['response_schemas_in_spec']}")
    print()

    if results["unmapped_schemas"]:
        print(f"UNMAPPED SCHEMAS ({len(results['unmapped_schemas'])}):")
        for item in results["unmapped_schemas"]:
            print(f"  {item['spec_schema']} — no Pydantic model and not in INTENTIONALLY_SKIPPED")
        print()

    drift = results["missing_fields"] + results["extra_fields"] + results["stale_mappings"]
    if drift:
        print(f"FIELD DRIFT ({len(drift)}):")
        for item in results["stale_mappings"]:
            error = item.get("error", "not in spec")
            print(f"  STALE MAPPING: {item['spec_schema']} -> {item['pydantic_model']} ({error})")
        for item in results["missing_fields"]:
            print(f"  MISSING in Pydantic: {item['spec_schema']}.{item['field']} not in {item['pydantic_model']}")
        for item in results["extra_fields"]:
            print(f"  EXTRA in Pydantic:   {item['pydantic_model']}.{item['field']} not in spec {item['spec_schema']}")
        print()

    if results["streaming_event_gaps"]:
        print(f"STREAMING EVENT GAPS ({len(results['streaming_event_gaps'])}):")
        for item in results["streaming_event_gaps"]:
            print(f"  UNCOVERED: {item['spec_schema']} has no Pydantic model and is not marked as skipped")
        print()

    if verbose and results["intentionally_skipped"]:
        print(f"INTENTIONALLY SKIPPED ({len(results['intentionally_skipped'])}):")
        for item in results["intentionally_skipped"]:
            print(f"  SKIPPED: {item['spec_schema']} — {item['reason']}")
        print()

    if s["total_issues"] == 0:
        print("OK: No unintentional drift detected.")
    else:
        print(f"TOTAL: {s['total_issues']} issue(s) found.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAI Responses schema drift")
    parser.add_argument("--spec", help="Path to OpenAI spec YAML file")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any drift")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show intentional divergences too")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write JSON report to docs/static/openai-responses-drift.json",
    )
    args = parser.parse_args()

    spec_path = _find_spec_file(args.spec)
    results = _run_checks(spec_path)

    _print_report(results, verbose=args.verbose)

    if args.update:
        output_path = REPO_ROOT / "docs" / "static" / "openai-responses-drift.json"
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nReport written to {output_path.relative_to(REPO_ROOT)}")

    if args.strict and results["summary"]["total_issues"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
