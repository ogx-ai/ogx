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

import io
import logging  # allow-direct-logging

import pytest
import structlog.contextvars
import structlog.stdlib

from ogx.log import _configure_structlog, _reset_logging_state, get_logger

# Keys that must always be redacted in log output.
# Keep in sync with SENSITIVE_LOG_KEYS — adding/removing here requires
# the same change in the production constant.
_SENSITIVE_TEST_KEYS = frozenset(
    [
        # User input
        "prompt",
        "messages",
        "message",
        "input_messages",
        "input_items",
        # Model output
        "content",
        "text",
        "final_response",
        "final_chat_completion",
        "output_messages",
        "output_items",
        "reasoning_content",
        # Model IDs (user choices)
        "model",
        "model_id",
        "provider_model_id",
        "provider_resource_id",
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
        "server_url",
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

    def test_contextvars_sensitive_key_is_redacted(self, caplog):
        """Sensitive values injected via structlog contextvars must be redacted.

        Regression: merge_contextvars runs before _redact_sensitive_keys, so
        context variables with sensitive keys cannot bypass redaction.
        """
        with caplog.at_level(logging.DEBUG):
            structlog.contextvars.bind_contextvars(model="context-leak-test")
            logger = get_logger("test.redaction", category="core")
            logger.info("test contextvars")
            output = caplog.text
            structlog.contextvars.clear_contextvars()
        assert "context-leak-test" not in output, (
            "Sensitive value injected via merge_contextvars leaked. "
            "_redact_sensitive_keys must run AFTER merge_contextvars."
        )
        assert "'model': '<REDACTED>'" in output

    def test_contextvars_sensitive_data_redacted_end_to_end(self, caplog):
        """End-to-end regression: sensitive data injected via merge_contextvars is redacted.

        Binds a sensitive context variable, logs through the structlog BoundLogger,
        and asserts the actual log output contains no leaked values.  Exercises
        the processor chain from contextvar binding through final rendering.
        """
        with caplog.at_level(logging.DEBUG):
            structlog.contextvars.bind_contextvars(model="contextvars-leak-test")
            logger = get_logger("test.redaction.cvx", category="core")
            logger.info("contextvars message", safe_key="keep_me")
            output = caplog.text
            structlog.contextvars.clear_contextvars()

        assert "contextvars-leak-test" not in output, (
            "Sensitive value injected via merge_contextvars leaked. "
            "_redact_sensitive_keys must run AFTER merge_contextvars in the chain."
        )
        assert "'model': '<REDACTED>'" in output
        assert "'safe_key': 'keep_me'" in output

    def test_stdlib_extra_sensitive_data_redacted_end_to_end(self, monkeypatch):
        """End-to-end regression: sensitive data injected via ExtraAdder is redacted.

        Calls the stdlib logger directly with extra= kwargs, which triggers
        ExtraAdder to copy those values into the event dict.  Uses a custom
        handler with the ProcessorFormatter to capture the structlog-processed
        output (caplog captures at the stdlib level, before the ProcessorFormatter
        applies foreign_pre_chain).

        Verifies that when ExtraAdder injects data into the event dict,
        _redact_sensitive_keys (which runs after ExtraAdder in the chain)
        catches those injected values.
        """
        _configure_structlog()
        shared_processors = _configure_structlog._shared_processors  # type: ignore[attr-defined]

        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )

        sink = io.StringIO()
        handler = logging.StreamHandler(sink)
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)

        logger = logging.getLogger("test.redaction.extra")
        old_handlers = logger.handlers[:]
        old_level = logger.level
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        try:
            logger.info(
                "extra= message",
                extra=dict(vector_store_id="extra-leak-test", safe_key="keep_me"),
            )
            processed = sink.getvalue()

            for leaked in ("extra-leak-test",):
                assert leaked not in processed, (
                    "Sensitive value injected via ExtraAdder leaked. "
                    "_redact_sensitive_keys must run AFTER ExtraAdder in the chain."
                )
            assert '"vector_store_id": "<REDACTED>"' in processed
            assert '"safe_key": "keep_me"' in processed
        finally:
            logger.handlers = old_handlers
            logger.level = old_level
            logger.propagate = True

    def test_test_keys_match_production_keys(self):
        """Both lists must be in sync.

        If a key is added to SENSITIVE_LOG_KEYS in production, this test fails
        until the corresponding entry is added to _SENSITIVE_TEST_KEYS.

        If a key is removed from SENSITIVE_LOG_KEYS, the parametrized
        test_sensitive_key_is_redacted for that key will fail because the
        value appears in log output instead of '<REDACTED>'.
        """
        from ogx.log import SENSITIVE_LOG_KEYS

        prod_only = SENSITIVE_LOG_KEYS - _SENSITIVE_TEST_KEYS
        test_only = _SENSITIVE_TEST_KEYS - SENSITIVE_LOG_KEYS

        if prod_only:
            pytest.fail(
                f"Production has {len(prod_only)} key(s) not in the test list: "
                f"{sorted(prod_only)}. Add them to _SENSITIVE_TEST_KEYS."
            )
        if test_only:
            pytest.fail(
                f"Test list has {len(test_only)} key(s) not in production: "
                f"{sorted(test_only)}. Remove them from _SENSITIVE_TEST_KEYS."
            )
