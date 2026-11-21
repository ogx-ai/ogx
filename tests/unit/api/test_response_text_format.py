# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAIResponseTextFormat is a discriminated union validated at parse time, so an
incomplete text.format is rejected before any provider work runs (#7007).
"""

import pytest
from pydantic import ValidationError

from ogx_api.openai_responses import (
    LENIENT_STORED_TEXT_FORMAT_CONTEXT_KEY,
    OpenAIResponseText,
    OpenAIResponseTextFormatJSONObject,
    OpenAIResponseTextFormatJSONSchema,
    OpenAIResponseTextFormatText,
)
from ogx_api.responses.models import CreateResponseRequest

MALFORMED_TEXT_FORMATS = [
    pytest.param({"name": "test_schema", "schema": {"type": "object"}}, id="missing-type"),
    pytest.param({"type": "bogus"}, id="unknown-type"),
    pytest.param({"type": "json_schema", "schema": {"type": "object"}}, id="missing-name"),
    pytest.param({"type": "json_schema", "name": "", "schema": {"type": "object"}}, id="empty-name"),
    pytest.param({"type": "json_schema", "name": "test_schema"}, id="missing-schema"),
    pytest.param({"type": "json_schema", "name": "test_schema", "schema": None}, id="null-schema"),
]


@pytest.mark.parametrize("text_format", MALFORMED_TEXT_FORMATS)
def test_malformed_text_format_is_rejected_at_parse_time(text_format):
    with pytest.raises(ValidationError):
        CreateResponseRequest(input="hi", model="test-model", text={"format": text_format})


@pytest.mark.parametrize(
    "text_format, expected_type",
    [
        ({"type": "text"}, OpenAIResponseTextFormatText),
        ({"type": "json_object"}, OpenAIResponseTextFormatJSONObject),
        ({"type": "json_schema", "name": "s", "schema": {"type": "object"}}, OpenAIResponseTextFormatJSONSchema),
    ],
)
def test_valid_text_format_parses_to_its_variant(text_format, expected_type):
    request = CreateResponseRequest(input="hi", model="test-model", text={"format": text_format})

    assert isinstance(request.text.format, expected_type)


@pytest.mark.parametrize("text", [None, {}, {"format": None}])
def test_text_omitted_or_null_is_still_accepted(text):
    payload = {"input": "hi", "model": "test-model"}
    if text is not None:
        payload["text"] = text

    request = CreateResponseRequest(**payload)

    assert request.text is None or request.text.format is None


@pytest.mark.parametrize(
    "text_format",
    [
        {"type": "text"},
        {"type": "json_object"},
        {"type": "json_schema", "name": "s", "schema": {"type": "object"}},
        {"type": "json_schema", "name": "s", "schema": {"type": "object"}, "description": "d", "strict": True},
    ],
)
def test_valid_text_format_dumps_exactly_what_was_sent(text_format):
    """No new explicit nulls on the wire: unset optional fields stay out of every dump."""
    text = OpenAIResponseText(format=text_format)

    assert text.model_dump()["format"] == text_format
    assert text.model_dump(mode="json")["format"] == text_format
    assert OpenAIResponseText.model_validate_json(text.model_dump_json()) == text


def test_json_schema_variant_accepts_the_schema_wire_key_by_name():
    text_format = OpenAIResponseTextFormatJSONSchema(name="s", schema={"type": "object"})

    assert text_format.schema_ == {"type": "object"}
    assert text_format.model_dump() == {"type": "json_schema", "name": "s", "schema": {"type": "object"}}


class TestLenientStoredTextFormat:
    """Responses stored before parse-time validation can carry an incomplete format;
    only the store-read path passes the lenient context, and reads it back as None.
    """

    @pytest.mark.parametrize("text_format", MALFORMED_TEXT_FORMATS)
    def test_legacy_format_reads_back_as_none_with_the_lenient_context(self, text_format):
        text = OpenAIResponseText.model_validate(
            {"format": text_format}, context={LENIENT_STORED_TEXT_FORMAT_CONTEXT_KEY: True}
        )

        assert text.format is None

    def test_valid_format_is_unchanged_with_the_lenient_context(self):
        text_format = {"type": "json_schema", "name": "s", "schema": {"type": "object"}}

        text = OpenAIResponseText.model_validate(
            {"format": text_format}, context={LENIENT_STORED_TEXT_FORMAT_CONTEXT_KEY: True}
        )

        assert isinstance(text.format, OpenAIResponseTextFormatJSONSchema)

    def test_legacy_format_is_still_rejected_without_the_context(self):
        with pytest.raises(ValidationError):
            OpenAIResponseText.model_validate({"format": {"type": "json_schema", "name": "s", "schema": None}})
