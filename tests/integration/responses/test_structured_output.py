# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import pytest


def test_json_schema_string_types(responses_client, text_model_id):
    """Test json schema with string types."""
    text_format = {
        "type": "json_schema",
        "name": "PersonProfile",
        "description": "A profile with multiple string fields",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "occupation": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["name", "occupation", "city"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate a profile for Tom, a 30-year-old software engineer in Raleigh",
        stream=False,
        text={"format": text_format},
    )
    assert response.text.format.model_dump(exclude_none=True, by_alias=True) == text_format

    data = json.loads(response.output_text)
    assert isinstance(data, dict)
    assert all(k in data and isinstance(data[k], str) for k in text_format["schema"]["properties"].keys())
    assert len(data) == 3


def test_json_schema_integer_types(responses_client, text_model_id):
    """Test json schema with integer types."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX model does not reliably produce valid structured JSON output")
    text_format = {
        "type": "json_schema",
        "name": "PersonAge",
        "description": "A person with an integer age field",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate a profile for Bob who is 25 years old",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["age"], int)


def test_json_schema_boolean_types(responses_client, text_model_id):
    """Test json schema with boolean types."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX model does not reliably produce valid structured JSON output")
    text_format = {
        "type": "json_schema",
        "name": "UserStatus",
        "description": "A user with boolean status fields",
        "schema": {
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "is_active": {"type": "boolean"},
                "email_verified": {"type": "boolean"},
            },
            "required": ["username", "is_active", "email_verified"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate a user profile for Alice who is active with verified email",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["is_active"], bool)
    assert isinstance(data["email_verified"], bool)


def test_json_schema_float_types(responses_client, text_model_id):
    """Test json schema with float types."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX model does not reliably produce valid structured JSON output")
    text_format = {
        "type": "json_schema",
        "name": "ProductPrice",
        "description": "A product with price information",
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
            },
            "required": ["product_name", "price"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate product info for a laptop priced at $999.99",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["price"], float)


def test_json_schema_array_of_strings(responses_client, text_model_id):
    """Test json schema with an array of strings."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX model does not reliably produce valid structured JSON output")
    text_format = {
        "type": "json_schema",
        "name": "PersonSkills",
        "description": "A person with a list of skills",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "skills"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate a profile for a developer named Charlie with skills: Python, JavaScript, and Docker",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["skills"], list)
    assert len(data["skills"]) > 0
    assert all(isinstance(skill, str) for skill in data["skills"])


def test_json_schema_array_of_integers(responses_client, text_model_id):
    """Test json schema with an array of integers."""
    text_format = {
        "type": "json_schema",
        "name": "TestScores",
        "description": "Test scores for a student",
        "schema": {
            "type": "object",
            "properties": {
                "student_name": {"type": "string"},
                "scores": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["student_name", "scores"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate test scores for student Dana with three test scores: 85, 92, and 78",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) > 0
    assert all(isinstance(score, int) for score in data["scores"])


def test_json_schema_array_of_objects(responses_client, text_model_id):
    """Test json schema with array of objects."""
    text_format = {
        "type": "json_schema",
        "name": "TeamMembers",
        "description": "A team with multiple members",
        "schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string"},
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                        },
                        "required": ["name", "role"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["team_name", "members"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate a team called Engineering with two members: Alice as lead and Bob as developer",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["members"], list)
    assert len(data["members"]) > 0
    assert all(
        isinstance(member, dict)
        and "name" in member
        and "role" in member
        and isinstance(member["name"], str)
        and isinstance(member["role"], str)
        for member in data["members"]
    )


def test_json_schema_nested_objects(responses_client, text_model_id):
    """Test json schema with nested object structures."""
    text_format = {
        "type": "json_schema",
        "name": "EmployeeRecord",
        "description": "An employee with nested department information",
        "schema": {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "employee_id": {"type": "integer"},
                    },
                    "required": ["name", "employee_id"],
                    "additionalProperties": False,
                },
                "department": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "manager": {"type": "string"},
                    },
                    "required": ["name", "manager"],
                    "additionalProperties": False,
                },
            },
            "required": ["employee", "department"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate an employee record for Susan (ID 1001) in Engineering department managed by Frank",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)
    assert isinstance(data["employee"], dict)
    assert isinstance(data["department"], dict)

    assert "name" in data["employee"]
    assert "employee_id" in data["employee"]
    assert isinstance(data["employee"]["name"], str)
    assert isinstance(data["employee"]["employee_id"], int)

    assert "name" in data["department"]
    assert "manager" in data["department"]
    assert isinstance(data["department"]["name"], str)
    assert isinstance(data["department"]["manager"], str)


def test_json_schema_mixed_types_structures(responses_client, text_model_id):
    """Test json schema with mixed types and structures."""
    text_format = {
        "type": "json_schema",
        "name": "ComplexProfile",
        "description": "Complex profile with mixed types",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "salary": {"type": "number"},
                "is_active": {"type": "boolean"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zipcode": {"type": "integer"},
                    },
                    "required": ["street", "city", "zipcode"],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "age", "salary", "is_active", "skills", "address"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = responses_client.responses.create(
        model=text_model_id,
        input="Generate a profile for Grace, age 35, salary $120000, active status true, with skills Python and SQL, living at 123 Main St in Raleigh, zipcode 27601",
        stream=False,
        text={"format": text_format},
    )

    data = json.loads(response.output_text)

    assert isinstance(data["name"], str)
    assert isinstance(data["age"], int)
    assert isinstance(data["salary"], int | float)
    assert isinstance(data["is_active"], bool)

    assert isinstance(data["skills"], list)
    assert all(isinstance(skill, str) for skill in data["skills"])

    assert isinstance(data["address"], dict)
    assert isinstance(data["address"]["street"], str)
    assert isinstance(data["address"]["city"], str)
    assert isinstance(data["address"]["zipcode"], int)

    assert set(data.keys()) == {"name", "age", "salary", "is_active", "skills", "address"}
    assert set(data["address"].keys()) == {"street", "city", "zipcode"}
