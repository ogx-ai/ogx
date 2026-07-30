# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from locust import HttpUser, task


class ChatCompletionUser(HttpUser):
    """
    Locust user for straightline performance testing of the OGX server.

    Sends chat completion requests continuously without delays to measure
    maximum throughput and minimal latency of the inference API.
    """

    def on_start(self):
        """Initialize user with model ID."""
        self.model_id = "openai/mock-chat-model"
        self.headers = {
            "Content-Type": "application/json",
        }

    @task
    def chat_completion(self):
        """Send a non-streaming chat completion request."""
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                },
            ],
            "max_tokens": 50,
        }

        with self.client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Invalid response format: missing choices")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
