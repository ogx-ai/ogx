# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.utils.config import redact_sensitive_fields


def test_redact_sensitive_fields_redacts_header_credentials():
    config = {
        "providers": {
            "responses": [
                {
                    "config": {
                        "moderation_headers": {
                            "Authorization": "Bearer sk-test",
                            "Cookie": "session=secret",
                            "Ocp-Apim-Subscription-Key": "subscription-secret",
                            "X-Api-Key": "secret",
                            "X-Trace-Id": "trace",
                        },
                    },
                }
            ]
        }
    }

    redacted = redact_sensitive_fields(config)

    assert redacted["providers"]["responses"][0]["config"]["moderation_headers"] == "********"


def test_redact_sensitive_fields_preserves_safe_token_fields():
    config = {
        "chunk_size_tokens": 512,
        "max_tokens": 1024,
        "api-token": "secret",
    }

    redacted = redact_sensitive_fields(config)

    assert redacted["chunk_size_tokens"] == 512
    assert redacted["max_tokens"] == 1024
    assert redacted["api-token"] == "********"
