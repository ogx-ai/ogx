# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os

import pytest

from ogx_api import ViolationLevel

VISION_SHIELD_ENABLED_PROVIDERS = {"together"}


def data_url_from_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{encoded_string}"
    return data_url


def test_safety_with_image(ogx_client, model_providers):
    vision_safety_enabled = len(VISION_SHIELD_ENABLED_PROVIDERS & model_providers) > 0
    if not vision_safety_enabled:
        pytest.skip(f"Testing vision shields is not supported for model_providers {model_providers}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    unsafe_example_img_path = f"{current_dir}/resources/example_unsafe.jpg"
    safe_example_img_path = f"{current_dir}/resources/example_safe.jpg"

    examples = {
        "safe": safe_example_img_path,
        "unsafe": unsafe_example_img_path,
    }

    for _, file_path in examples.items():
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tell me how to assemble this",
                },
                {
                    "type": "image",
                    "image": {"url": {"uri": data_url_from_image(file_path)}},
                },
            ],
        }
        response = ogx_client.safety.run_shield(
            messages=[message],
            shield_id="meta-llama/Llama-Guard-3-11B-Vision",
        )
        assert response is not None

        # FIXME: We are getting flaky results with the unsafe example:
        #   1. sometimes it is marked as safe
        #   2. sometimes it is marked as unsafe but with incorrect violation_type
        #   3. sometimes it is marked as unsafe with correct violation_type
        if response.violation is not None:
            assert response.violation.violation_level == ViolationLevel.ERROR.value
            assert response.violation.user_message == "I can't answer that. Can I help with something else?"
