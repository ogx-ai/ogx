# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for OpenAI compatibility tool conversion.
Tests convert_tooldef_to_openai_tool with new JSON Schema approach.
"""

from ogx.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool


class TestSimpleSchemaConversion:
    """Test basic schema conversions to OpenAI format."""

    def test_simple_tool_conversion(self):
        """Test conversion of simple tool with basic input schema."""
        input_schema = {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        }

        result = convert_tooldef_to_openai_tool(
            tool_name="get_weather",
            description="Get weather information",
            input_schema=input_schema,
        )

        # Check OpenAI structure
        assert result["type"] == "function"
        assert "function" in result

        function = result["function"]
        assert function["name"] == "get_weather"
        assert function["description"] == "Get weather information"

        # Check parameters are passed through
        assert "parameters" in function
        assert function["parameters"] == input_schema
        assert function["parameters"]["type"] == "object"
        assert "location" in function["parameters"]["properties"]

    def test_tool_without_description(self):
        """Test tool conversion without description."""
        result = convert_tooldef_to_openai_tool(
            tool_name="test_tool",
            input_schema={"type": "object", "properties": {}},
        )

        assert result["function"]["name"] == "test_tool"
        assert "description" not in result["function"]
        assert "parameters" in result["function"]


class TestComplexSchemaConversion:
    """Test conversion of complex JSON Schema features."""

    def test_schema_with_refs_and_defs(self):
        """Test that $ref and $defs are passed through to OpenAI."""
        result = convert_tooldef_to_openai_tool(
            tool_name="book_flight",
            description="Book a flight",
            input_schema={
                "type": "object",
                "properties": {
                    "flight": {"$ref": "#/$defs/FlightInfo"},
                    "passengers": {"type": "array", "items": {"$ref": "#/$defs/Passenger"}},
                    "payment": {"$ref": "#/$defs/Payment"},
                },
                "required": ["flight", "passengers", "payment"],
                "$defs": {
                    "FlightInfo": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string", "description": "Departure airport"},
                            "to": {"type": "string", "description": "Arrival airport"},
                            "date": {"type": "string", "format": "date"},
                        },
                        "required": ["from", "to", "date"],
                    },
                    "Passenger": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
                        "required": ["name", "age"],
                    },
                    "Payment": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["credit_card", "debit_card"]},
                            "amount": {"type": "number", "minimum": 0},
                        },
                    },
                },
            },
        )

        params = result["function"]["parameters"]

        # Verify $defs are preserved
        assert "$defs" in params
        assert "FlightInfo" in params["$defs"]
        assert "Passenger" in params["$defs"]
        assert "Payment" in params["$defs"]

        # Verify $ref are preserved
        assert params["properties"]["flight"]["$ref"] == "#/$defs/FlightInfo"
        assert params["properties"]["passengers"]["items"]["$ref"] == "#/$defs/Passenger"
        assert params["properties"]["payment"]["$ref"] == "#/$defs/Payment"

        # Verify nested schema details are preserved
        assert params["$defs"]["FlightInfo"]["properties"]["date"]["format"] == "date"
        assert params["$defs"]["Passenger"]["properties"]["age"]["minimum"] == 0
        assert params["$defs"]["Payment"]["properties"]["method"]["enum"] == ["credit_card", "debit_card"]

    def test_anyof_schema_conversion(self):
        """Test conversion of anyOf schemas."""
        result = convert_tooldef_to_openai_tool(
            tool_name="flexible_input",
            input_schema={
                "type": "object",
                "properties": {
                    "contact": {
                        "anyOf": [
                            {"type": "string", "format": "email"},
                            {"type": "string", "pattern": "^\\+?[0-9]{10,15}$"},
                        ],
                        "description": "Email or phone number",
                    }
                },
            },
        )

        contact_schema = result["function"]["parameters"]["properties"]["contact"]
        assert "anyOf" in contact_schema
        assert len(contact_schema["anyOf"]) == 2
        assert contact_schema["anyOf"][0]["format"] == "email"
        assert "pattern" in contact_schema["anyOf"][1]

    def test_nested_objects_conversion(self):
        """Test conversion of deeply nested objects."""
        result = convert_tooldef_to_openai_tool(
            tool_name="nested_data",
            input_schema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "profile": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "settings": {
                                        "type": "object",
                                        "properties": {"theme": {"type": "string", "enum": ["light", "dark"]}},
                                    },
                                },
                            }
                        },
                    }
                },
            },
        )

        # Navigate deep structure
        user_schema = result["function"]["parameters"]["properties"]["user"]
        profile_schema = user_schema["properties"]["profile"]
        settings_schema = profile_schema["properties"]["settings"]
        theme_schema = settings_schema["properties"]["theme"]

        assert theme_schema["enum"] == ["light", "dark"]

    def test_array_schemas_with_constraints(self):
        """Test conversion of array schemas with constraints."""
        result = convert_tooldef_to_openai_tool(
            tool_name="list_processor",
            input_schema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                            "required": ["id"],
                        },
                        "minItems": 1,
                        "maxItems": 100,
                        "uniqueItems": True,
                    }
                },
            },
        )

        items_schema = result["function"]["parameters"]["properties"]["items"]
        assert items_schema["type"] == "array"
        assert items_schema["minItems"] == 1
        assert items_schema["maxItems"] == 100
        assert items_schema["uniqueItems"] is True
        assert items_schema["items"]["type"] == "object"


class TestOutputSchemaHandling:
    """Test that output_schema is correctly handled (or dropped) for OpenAI."""

    def test_output_schema_is_dropped(self):
        """Test that output_schema is NOT included in OpenAI format (API limitation)."""
        input_schema = {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}

        result = convert_tooldef_to_openai_tool(
            tool_name="calculator",
            description="Perform calculation",
            input_schema=input_schema,
        )

        # OpenAI doesn't support output schema
        assert "outputSchema" not in result["function"]
        assert "responseSchema" not in result["function"]
        assert "output_schema" not in result["function"]

        # But input schema should be present
        assert "parameters" in result["function"]
        assert result["function"]["parameters"] == input_schema

    def test_only_output_schema_no_input(self):
        """Test tool with only output_schema (unusual but valid)."""
        result = convert_tooldef_to_openai_tool(
            tool_name="no_input_tool",
            description="Tool with no inputs",
        )

        # No parameters should be set if input_schema is None
        # (or we might set an empty object schema - implementation detail)
        assert "outputSchema" not in result["function"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_tool_with_no_schemas(self):
        """Test tool with neither input nor output schema."""
        result = convert_tooldef_to_openai_tool(
            tool_name="schemaless_tool",
            description="Tool without schemas",
        )

        assert result["function"]["name"] == "schemaless_tool"
        assert result["function"]["description"] == "Tool without schemas"
        # Implementation detail: might have no parameters or empty object

    def test_empty_input_schema(self):
        """Test tool with empty object schema."""
        result = convert_tooldef_to_openai_tool(
            tool_name="no_params",
            input_schema={"type": "object", "properties": {}},
        )

        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}

    def test_schema_with_additional_properties(self):
        """Test that additionalProperties is preserved."""
        result = convert_tooldef_to_openai_tool(
            tool_name="flexible_tool",
            input_schema={
                "type": "object",
                "properties": {"known_field": {"type": "string"}},
                "additionalProperties": True,
            },
        )

        assert result["function"]["parameters"]["additionalProperties"] is True

    def test_schema_with_pattern_properties(self):
        """Test that patternProperties is preserved."""
        result = convert_tooldef_to_openai_tool(
            tool_name="pattern_tool",
            input_schema={"type": "object", "patternProperties": {"^[a-z]+$": {"type": "string"}}},
        )

        assert "patternProperties" in result["function"]["parameters"]

    def test_schema_identity(self):
        """Test that converted schema is identical to input (no lossy conversion)."""
        original_schema = {
            "type": "object",
            "properties": {"complex": {"$ref": "#/$defs/Complex"}},
            "$defs": {
                "Complex": {
                    "type": "object",
                    "properties": {"nested": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
                }
            },
            "required": ["complex"],
            "additionalProperties": False,
        }

        result = convert_tooldef_to_openai_tool(
            tool_name="test",
            input_schema=original_schema,
        )

        # Converted parameters should be EXACTLY the same as input
        assert result["function"]["parameters"] == original_schema


class TestConversionConsistency:
    """Test consistency across multiple conversions."""

    def test_multiple_tools_with_shared_defs(self):
        """Test converting multiple tools that could share definitions."""
        result1 = convert_tooldef_to_openai_tool(
            tool_name="tool1",
            input_schema={
                "type": "object",
                "properties": {"data": {"$ref": "#/$defs/Data"}},
                "$defs": {"Data": {"type": "object", "properties": {"x": {"type": "number"}}}},
            },
        )

        result2 = convert_tooldef_to_openai_tool(
            tool_name="tool2",
            input_schema={
                "type": "object",
                "properties": {"info": {"$ref": "#/$defs/Data"}},
                "$defs": {"Data": {"type": "object", "properties": {"y": {"type": "string"}}}},
            },
        )

        # Each tool maintains its own $defs independently
        assert result1["function"]["parameters"]["$defs"]["Data"]["properties"]["x"]["type"] == "number"
        assert result2["function"]["parameters"]["$defs"]["Data"]["properties"]["y"]["type"] == "string"

    def test_conversion_is_pure(self):
        """Test that conversion doesn't modify the original input."""
        original_schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "$defs": {"T": {"type": "number"}},
        }

        # Convert
        result = convert_tooldef_to_openai_tool(
            tool_name="test",
            input_schema=original_schema.copy(),
        )

        # Original schema should be unchanged
        assert original_schema == {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "$defs": {"T": {"type": "number"}},
        }
        assert "$defs" in original_schema
        # Verify result has the expected structure
        assert result["function"]["parameters"] == original_schema
