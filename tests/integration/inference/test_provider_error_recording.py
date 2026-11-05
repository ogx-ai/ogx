# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration test demonstrating provider error recording and replay.

This test verifies the core feature of PR #4880: when a provider SDK raises an
exception during a live API call (record mode), the recording system serializes
the exception. On subsequent runs (replay mode), the same exception is
deserialized and re-raised so the test passes identically without a live
provider connection.

Run in record mode (requires a live provider, e.g. Ollama or OpenAI):
    pytest tests/integration/inference/test_provider_error_recording.py \\
        --setup=ollama --stack-config=server:starter --inference-mode=record

Run in replay mode (uses committed recording, no live provider needed):
    pytest tests/integration/inference/test_provider_error_recording.py \\
        --setup=ollama --inference-mode=replay
"""

import pytest
from openai import BadRequestError


class TestProviderErrorRecording:
    """Verify that provider SDK exceptions survive the record -> replay cycle.

    In record mode, a real API call triggers a provider error that gets
    serialized to a recording file. In replay mode, the error is reconstructed
    from the recording so the test behaves identically without a live provider.
    """

    def test_provider_rejects_invalid_base64_image(self, openai_client, text_model_id):
        """A malformed base64 image triggers a provider-level error.

        This request passes through OGX to the inference provider, which
        rejects the invalid image data. The recording system captures this
        provider exception so that replay mode reproduces the exact same error.
        """
        # llama.cpp servers without an mmproj adapter reject all image input
        # (500 "image input is not supported"), so there is no invalid-image
        # rejection to record. TODO: remove this skip once the llama-cpp-server
        # lane serves an image-capable model, and re-record.
        if (text_model_id or "").startswith("llama-cpp-server/"):
            pytest.skip("llama-cpp-server does not support image input; nothing to record")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=text_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,not_valid_base64_data!!!"},
                            },
                        ],
                    }
                ],
            )

        assert exc_info.value.status_code == 400
