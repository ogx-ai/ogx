# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests that id and status fields are always str (never None) on response models."""

import pytest
from pydantic import ValidationError

from llama_stack_api import (
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageFunctionToolCall,
)


class TestOpenAIResponseMessageFields:
    def test_default_id_is_str(self):
        msg = OpenAIResponseMessage(content="hello", role="user")
        assert isinstance(msg.id, str)
        assert msg.id.startswith("msg_")

    def test_default_status_is_str(self):
        msg = OpenAIResponseMessage(content="hello", role="user")
        assert isinstance(msg.status, str)
        assert msg.status == "completed"

    def test_explicit_id_and_status(self):
        msg = OpenAIResponseMessage(content="hello", role="user", id="custom_id", status="in_progress")
        assert msg.id == "custom_id"
        assert msg.status == "in_progress"

    def test_two_messages_get_different_ids(self):
        msg1 = OpenAIResponseMessage(content="a", role="user")
        msg2 = OpenAIResponseMessage(content="b", role="user")
        assert msg1.id != msg2.id

    def test_none_id_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIResponseMessage(content="hello", role="user", id=None)

    def test_none_status_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIResponseMessage(content="hello", role="user", status=None)


class TestOpenAIResponseOutputMessageFunctionToolCallFields:
    def test_default_id_is_str(self):
        tc = OpenAIResponseOutputMessageFunctionToolCall(call_id="call_123", name="func", arguments="{}")
        assert isinstance(tc.id, str)
        assert tc.id.startswith("fc_")

    def test_default_status_is_str(self):
        tc = OpenAIResponseOutputMessageFunctionToolCall(call_id="call_123", name="func", arguments="{}")
        assert isinstance(tc.status, str)
        assert tc.status == "in_progress"

    def test_explicit_id_and_status(self):
        tc = OpenAIResponseOutputMessageFunctionToolCall(
            call_id="call_123", name="func", arguments="{}", id="my_id", status="completed"
        )
        assert tc.id == "my_id"
        assert tc.status == "completed"

    def test_none_id_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIResponseOutputMessageFunctionToolCall(call_id="call_123", name="func", arguments="{}", id=None)

    def test_none_status_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIResponseOutputMessageFunctionToolCall(call_id="call_123", name="func", arguments="{}", status=None)


class TestOpenAIResponseInputFunctionToolCallOutputFields:
    def test_default_id_is_str(self):
        fco = OpenAIResponseInputFunctionToolCallOutput(call_id="call_123", output="result")
        assert isinstance(fco.id, str)
        assert fco.id.startswith("fco_")

    def test_default_status_is_str(self):
        fco = OpenAIResponseInputFunctionToolCallOutput(call_id="call_123", output="result")
        assert isinstance(fco.status, str)
        assert fco.status == "completed"

    def test_explicit_id_and_status(self):
        fco = OpenAIResponseInputFunctionToolCallOutput(
            call_id="call_123", output="result", id="my_id", status="in_progress"
        )
        assert fco.id == "my_id"
        assert fco.status == "in_progress"

    def test_none_id_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIResponseInputFunctionToolCallOutput(call_id="call_123", output="result", id=None)

    def test_none_status_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIResponseInputFunctionToolCallOutput(call_id="call_123", output="result", status=None)
