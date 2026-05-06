# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import pathlib

import pytest


@pytest.fixture
def image_path():
    return pathlib.Path(__file__).parent / "dog.png"


@pytest.fixture
def base64_image_data(image_path):
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


async def test_openai_chat_completion_image_url(openai_client, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/meta-llama/ogx/main/tests/integration/inference/dog.png"
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }

    response = openai_client.chat.completions.create(
        model=vision_model_id,
        messages=[message],
        stream=False,
    )

    message_content = response.choices[0].message.content.lower().strip()
    assert len(message_content) > 0
    assert any(expected in message_content for expected in {"dog", "puppy", "pup"})


async def test_openai_chat_completion_image_data(openai_client, vision_model_id, base64_image_data):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image_data}",
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }

    response = openai_client.chat.completions.create(
        model=vision_model_id,
        messages=[message],
        stream=False,
    )

    message_content = response.choices[0].message.content.lower().strip()
    assert len(message_content) > 0
    assert any(expected in message_content for expected in {"dog", "puppy", "pup"})
