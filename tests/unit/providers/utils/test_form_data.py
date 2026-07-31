# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from ogx.providers.utils.files.form_data import (
    parse_expires_after,
    parse_pydantic_from_form,
)


class _TestModel(BaseModel):
    """Simple test model for generic parsing tests."""

    name: str
    value: int


async def test_parse_pydantic_from_form_bracket_notation():
    """Test parsing a Pydantic model using bracket notation."""
    # Create mock request with form data
    mock_request = MagicMock()
    mock_form = {
        "test_field[name]": "test_name",
        "test_field[value]": "42",
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is not None
    assert result.name == "test_name"
    assert result.value == 42


async def test_parse_pydantic_from_form_json_string():
    """Test parsing a Pydantic model from JSON string."""
    # Create mock request with form data
    mock_request = MagicMock()
    test_data = {"name": "test_name", "value": 42}
    mock_form = {
        "test_field": json.dumps(test_data),
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is not None
    assert result.name == "test_name"
    assert result.value == 42


async def test_parse_pydantic_from_form_bracket_takes_precedence():
    """Test that bracket notation takes precedence over JSON string."""
    # Create mock request with both formats
    mock_request = MagicMock()
    mock_form = {
        "test_field[name]": "bracket_name",
        "test_field[value]": "100",
        "test_field": json.dumps({"name": "json_name", "value": 50}),
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is not None
    # Bracket notation should win
    assert result.name == "bracket_name"
    assert result.value == 100


async def test_parse_pydantic_from_form_missing_field():
    """Test that None is returned when field is missing."""
    # Create mock request with empty form
    mock_request = MagicMock()
    mock_form = {}
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is None


async def test_parse_pydantic_from_form_invalid_json():
    """Test that None is returned for invalid JSON."""
    # Create mock request with invalid JSON
    mock_request = MagicMock()
    mock_form = {
        "test_field": "not valid json",
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is None


async def test_parse_pydantic_from_form_invalid_data():
    """Test that None is returned when data doesn't match model."""
    # Create mock request with data that doesn't match the model
    mock_request = MagicMock()
    mock_form = {
        "test_field[wrong_field]": "value",
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is None


async def test_parse_expires_after_bracket_notation():
    """Test parsing expires_after using bracket notation."""
    # Create mock request with form data
    mock_request = MagicMock()
    mock_form = {
        "expires_after[anchor]": "created_at",
        "expires_after[seconds]": "3600",
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_expires_after(mock_request)

    assert result is not None
    assert result.anchor == "created_at"
    assert result.seconds == 3600


async def test_parse_expires_after_json_string():
    """Test parsing expires_after from JSON string."""
    # Create mock request with form data
    mock_request = MagicMock()
    expires_data = {"anchor": "created_at", "seconds": 7200}
    mock_form = {
        "expires_after": json.dumps(expires_data),
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_expires_after(mock_request)

    assert result is not None
    assert result.anchor == "created_at"
    assert result.seconds == 7200


async def test_parse_expires_after_missing():
    """Test that None is returned when expires_after is missing."""
    # Create mock request with empty form
    mock_request = MagicMock()
    mock_form = {}
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_expires_after(mock_request)

    assert result is None


async def test_parse_pydantic_from_form_type_conversion():
    """Test that bracket notation properly handles type conversion."""
    # Create mock request with string values that need conversion
    mock_request = MagicMock()
    mock_form = {
        "test_field[name]": "test",
        "test_field[value]": "999",  # String that should be converted to int
    }
    mock_request.form = AsyncMock(return_value=mock_form)

    result = await parse_pydantic_from_form(mock_request, "test_field", _TestModel)

    assert result is not None
    assert result.name == "test"
    assert result.value == 999
    assert isinstance(result.value, int)
