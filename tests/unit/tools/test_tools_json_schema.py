# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for JSON Schema-based tool definitions.
Tests the new input_schema and output_schema fields.
"""

from pydantic import ValidationError

from ogx_api import ToolDef


class TestToolDefValidation:
    """Test ToolDef validation with JSON Schema."""

    def test_simple_input_schema(self):
        """Test ToolDef with simple input schema."""
        tool = ToolDef(
            name="get_weather",
            description="Get weather information",
            input_schema={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        )

        assert tool.name == "get_weather"
        assert tool.input_schema["type"] == "object"
        assert "location" in tool.input_schema["properties"]
        assert tool.output_schema is None

    def test_input_and_output_schema(self):
        """Test ToolDef with both input and output schemas."""
        tool = ToolDef(
            name="calculate",
            description="Perform calculation",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
            output_schema={"type": "object", "properties": {"result": {"type": "number"}}, "required": ["result"]},
        )

        assert tool.input_schema is not None
        assert tool.output_schema is not None
        assert "result" in tool.output_schema["properties"]

    def test_schema_with_refs_and_defs(self):
        """Test that $ref and $defs are preserved in schemas."""
        tool = ToolDef(
            name="book_flight",
            description="Book a flight",
            input_schema={
                "type": "object",
                "properties": {
                    "flight": {"$ref": "#/$defs/FlightInfo"},
                    "passengers": {"type": "array", "items": {"$ref": "#/$defs/Passenger"}},
                },
                "$defs": {
                    "FlightInfo": {
                        "type": "object",
                        "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
                    },
                    "Passenger": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    },
                },
            },
        )

        # Verify $defs are preserved
        assert "$defs" in tool.input_schema
        assert "FlightInfo" in tool.input_schema["$defs"]
        assert "Passenger" in tool.input_schema["$defs"]

        # Verify $ref are preserved
        assert tool.input_schema["properties"]["flight"]["$ref"] == "#/$defs/FlightInfo"
        assert tool.input_schema["properties"]["passengers"]["items"]["$ref"] == "#/$defs/Passenger"

    def test_output_schema_with_refs(self):
        """Test that output_schema also supports $ref and $defs."""
        tool = ToolDef(
            name="search",
            description="Search for items",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            output_schema={
                "type": "object",
                "properties": {"results": {"type": "array", "items": {"$ref": "#/$defs/SearchResult"}}},
                "$defs": {
                    "SearchResult": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}, "score": {"type": "number"}},
                    }
                },
            },
        )

        assert "$defs" in tool.output_schema
        assert "SearchResult" in tool.output_schema["$defs"]

    def test_complex_json_schema_features(self):
        """Test various JSON Schema features are preserved."""
        tool = ToolDef(
            name="complex_tool",
            description="Tool with complex schema",
            input_schema={
                "type": "object",
                "properties": {
                    # anyOf
                    "contact": {
                        "anyOf": [
                            {"type": "string", "format": "email"},
                            {"type": "string", "pattern": "^\\+?[0-9]{10,15}$"},
                        ]
                    },
                    # enum
                    "status": {"type": "string", "enum": ["pending", "approved", "rejected"]},
                    # nested objects
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "zipcode": {"type": "string", "pattern": "^[0-9]{5}$"},
                        },
                        "required": ["street", "city"],
                    },
                    # array with constraints
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10,
                        "uniqueItems": True,
                    },
                },
            },
        )

        # Verify anyOf
        assert "anyOf" in tool.input_schema["properties"]["contact"]

        # Verify enum
        assert tool.input_schema["properties"]["status"]["enum"] == ["pending", "approved", "rejected"]

        # Verify nested object
        assert tool.input_schema["properties"]["address"]["type"] == "object"
        assert "zipcode" in tool.input_schema["properties"]["address"]["properties"]

        # Verify array constraints
        tags_schema = tool.input_schema["properties"]["tags"]
        assert tags_schema["minItems"] == 1
        assert tags_schema["maxItems"] == 10
        assert tags_schema["uniqueItems"] is True

    def test_invalid_json_schema_raises_error(self):
        """Test that invalid JSON Schema raises validation error."""
        # TODO: This test will pass once we add schema validation
        # For now, Pydantic accepts any dict, so this is a placeholder

        # This should eventually raise an error due to invalid schema
        try:
            ToolDef(
                name="bad_tool",
                input_schema={
                    "type": "invalid_type",  # Not a valid JSON Schema type
                    "properties": "not_an_object",  # Should be an object
                },
            )
            # For now this passes, but shouldn't after we add validation
        except ValidationError:
            pass  # Expected once validation is added


class TestSchemaEquivalence:
    """Test that schemas remain unchanged through serialization."""

    def test_schema_roundtrip(self):
        """Test that schemas survive model_dump/model_validate roundtrip."""
        original = ToolDef(
            name="test",
            input_schema={
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/X"}},
                "$defs": {"X": {"type": "string"}},
            },
        )

        # Serialize and deserialize
        dumped = original.model_dump()
        restored = ToolDef(**dumped)

        # Schemas should be identical
        assert restored.input_schema == original.input_schema
        assert "$defs" in restored.input_schema
        assert restored.input_schema["properties"]["x"]["$ref"] == "#/$defs/X"

    def test_json_serialization(self):
        """Test JSON serialization preserves schema."""
        import json

        tool = ToolDef(
            name="test",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$defs": {"T": {"type": "number"}},
            },
            output_schema={"type": "object", "properties": {"b": {"$ref": "#/$defs/T"}}},
        )

        # Serialize to JSON and back
        json_str = tool.model_dump_json()
        parsed = json.loads(json_str)
        restored = ToolDef(**parsed)

        assert restored.input_schema == tool.input_schema
        assert restored.output_schema == tool.output_schema
        assert "$defs" in restored.input_schema


class TestBackwardsCompatibility:
    """Test handling of legacy code patterns."""

    def test_none_schemas(self):
        """Test tools with no schemas (legacy case)."""
        tool = ToolDef(name="legacy_tool", description="Tool without schemas", input_schema=None, output_schema=None)

        assert tool.input_schema is None
        assert tool.output_schema is None

    def test_metadata_preserved(self):
        """Test that metadata field still works."""
        tool = ToolDef(
            name="test", input_schema={"type": "object"}, metadata={"endpoint": "http://example.com", "version": "1.0"}
        )

        assert tool.metadata["endpoint"] == "http://example.com"
        assert tool.metadata["version"] == "1.0"
