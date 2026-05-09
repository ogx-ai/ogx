# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import pathlib
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).parent


@pytest.fixture
def image_path():
    return pathlib.Path(__file__).parent / "dog.png"


@pytest.fixture
def base64_image_data(image_path):
    # Convert the image to base64
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


@pytest.fixture
def base64_image_url(base64_image_data):
    return f"data:image/png;base64,{base64_image_data}"


def test_image_chat_completion_non_streaming(client_with_models, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/ogx-ai/ogx/main/tests/integration/inference/dog.png"
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = client_with_models.chat.completions.create(
        model=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.choices[0].message.content.lower().strip()
    assert len(message_content) > 0
    assert any(expected in message_content for expected in {"dog", "puppy", "pup"})


@pytest.fixture
def multi_image_data():
    files = [
        THIS_DIR / "vision_test_1.jpg",
        THIS_DIR / "vision_test_2.jpg",
        THIS_DIR / "vision_test_3.jpg",
    ]
    encoded_files = []
    for file in files:
        with open(file, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_files.append(base64_data)
    return encoded_files


@pytest.fixture
def multi_image_url(multi_image_data):
    return [f"data:image/jpeg;base64,{data}" for data in multi_image_data]


@pytest.mark.parametrize("stream", [True, False])
def test_image_chat_completion_multiple_images(client_with_models, vision_model_id, multi_image_url, stream):
    supported_models = ["llama-4", "gpt-4o", "llama4"]
    if not any(model in vision_model_id.lower() for model in supported_models):
        pytest.skip(
            f"Skip since multi-image tests are only supported for {supported_models}, not for {vision_model_id}"
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": multi_image_url[0],
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": multi_image_url[1],
                    },
                },
                {
                    "type": "text",
                    "text": "What are the differences between these images? Where would you assume they would be located?",
                },
            ],
        },
    ]
    response = client_with_models.chat.completions.create(
        model=vision_model_id,
        messages=messages,
        stream=stream,
    )
    if stream:
        message_content = ""
        for chunk in response:
            message_content += chunk.choices[0].delta.content
    else:
        message_content = response.choices[0].message.content
    assert len(message_content) > 0
    assert any(expected in message_content.lower().strip() for expected in {"bedroom"}), message_content

    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": message_content}],
            "stop_reason": "end_of_turn",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": multi_image_data[2],
                    },
                },
                {"type": "text", "text": "How about this one?"},
            ],
        },
    )
    response = client_with_models.chat.completions.create(
        model=vision_model_id,
        messages=messages,
        stream=stream,
    )
    if stream:
        message_content = ""
        for chunk in response:
            message_content += chunk.event.delta.text
    else:
        message_content = response.choices[0].message.content
    assert len(message_content) > 0
    assert any(expected in message_content.lower().strip() for expected in {"sword", "shield"}), message_content


def test_image_chat_completion_streaming(client_with_models, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://raw.githubusercontent.com/ogx-ai/ogx/main/tests/integration/inference/dog.png"
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = client_with_models.chat.completions.create(
        model=vision_model_id,
        messages=[message],
        stream=True,
    )
    streamed_content = ""
    for chunk in response:
        streamed_content += chunk.choices[0].delta.content.lower()
    assert len(streamed_content) > 0
    assert any(expected in streamed_content for expected in {"dog", "puppy", "pup"})


def test_image_chat_completion_base64(client_with_models, vision_model_id, base64_image_url):
    image_spec = {
        "type": "image_url",
        "image_url": {
            "url": base64_image_url,
        },
    }

    message = {
        "role": "user",
        "content": [
            image_spec,
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = client_with_models.chat.completions.create(
        model=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.choices[0].message.content.lower().strip()
    assert len(message_content) > 0
