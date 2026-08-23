# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Regression test: verify that sensitive values are redacted from structured log
output via the _redact_sensitive_keys structlog processor.

The keys below are hardcoded — they are NOT derived from SENSITIVE_LOG_KEYS.
If a developer removes a key from SENSITIVE_LOG_KEYS without updating the test,
the test will fail and catch the omission.  Do NOT tie this list to the
production constant.
"""

import logging  # allow-direct-logging

import pytest

from ogx.log import _reset_logging_state, get_logger

# Keys that must always be redacted in log output.
# Keep in sync with SENSITIVE_LOG_KEYS — adding/removing here requires
# the same change in the production constant.
_SENSITIVE_TEST_KEYS = frozenset(
    [
        # User input
        "prompt",
        "messages",
        "input_messages",
        "input_items",
        # Model output
        "content",
        "text",
        "final_response",
        "output_messages",
        "output_items",
        "reasoning_content",
        # Model IDs (user choices)
        "model",
        "model_id",
        "provider_model_id",
        # Session identifiers
        "conversation",
        "conversation_id",
        "batch_id",
        "request_id",
        "completion_id",
        "stream_id",
        # Tool data
        "tool_calls",
        "tool_call",
        "tool_name",
        "tool_call_id",
        # User data
        "url",
        "vector_store_id",
        "file_id",
        # Response / body
        "response",
        "body",
        "exc",
    ]
)


@pytest.fixture(autouse=True)
def _clean_logging_state():
    _reset_logging_state()
    yield
    _reset_logging_state()


class TestSensitiveDataRedactedFromLogOutput:
    """Verify that sensitive values are replaced with '<REDACTED>' in log output."""

    @pytest.mark.parametrize("key", sorted(_SENSITIVE_TEST_KEYS))
    def test_sensitive_key_is_redacted(self, key, caplog):
        """Each hardcoded sensitive key must appear as '<REDACTED>' in log output."""
        sensitive_value = f"sensitive_{key}"
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("test.redaction", category="core")
            logger.info("test message", **{key: sensitive_value})
        output = caplog.text
        assert sensitive_value not in output, (
            f"Log output leaked the value '{sensitive_value}' for key '{key}'. "
            f"This key must be redacted. Add '{key}' to SENSITIVE_LOG_KEYS "
            f"if it is not already there."
        )
        # Log output is a dict repr: {'key': '<REDACTED>', ...}
        assert f"'{key}': '<REDACTED>'" in output, (
            f"Log output for key '{key}' does not show '<REDACTED>'. Check the _redact_sensitive_keys processor."
        )

    @pytest.mark.parametrize(
        "level",
        [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ],
    )
    def test_sensitive_key_redacted_at_all_levels(self, level, caplog):
        """Sensitive keys must be redacted regardless of log level."""
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("test.redaction", category="core")
            # Ensure the stdlib logger allows through the level we're testing
            logger.setLevel(logging.DEBUG)
            logger.log(level, "test message", model="should-be-redacted")
        output = caplog.text
        assert "should-be-redacted" not in output, (
            f"Log output leaked the model name at level {logging.getLevelName(level)}."
        )
        assert "'model': '<REDACTED>'" in output

    @pytest.mark.parametrize(
        "value",
        [
            "",
            None,
            {"nested": "dict"},
            ["a", "list"],
            42,
        ],
    )
    def test_sensitive_key_redacted_for_all_value_types(self, value, caplog):
        """Sensitive keys must be redacted regardless of value type."""
        key = "prompt"
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("test.redaction", category="core")
            logger.info("test message", **{key: value})
        output = caplog.text
        if isinstance(value, str) and value:
            assert value not in output
        assert f"'{key}': '<REDACTED>'" in output

    def test_non_sensitive_keys_are_preserved(self, caplog):
        """Safe keys must appear with their original values."""
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("test.redaction", category="core")
            logger.info("test message", safe_key="keep_me", another_safe=42)
        output = caplog.text
        assert "'safe_key': 'keep_me'" in output
        assert "'another_safe': 42" in output

    def test_event_message_is_preserved(self, caplog):
        """The event message string must never be redacted."""
        with caplog.at_level(logging.DEBUG):
            logger = get_logger("test.redaction", category="core")
            logger.info("my event message", model="should-be-redacted")
        output = caplog.text
        assert "my event message" in output
        assert "'model': '<REDACTED>'" in output
