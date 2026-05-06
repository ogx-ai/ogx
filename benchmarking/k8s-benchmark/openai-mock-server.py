#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAI-compatible mock server that returns:
- Hardcoded /models response for consistent validation
- Valid OpenAI-formatted chat completion responses with dynamic content
"""

import argparse
import json
import os
import random
import time
import uuid

from flask import Flask, Response, jsonify, request

app = Flask(__name__)


# Models from environment variables
def get_models():
    models_str = os.getenv("MOCK_MODELS", "meta-llama/Llama-3.2-3B-Instruct")
    model_ids = [m.strip() for m in models_str.split(",") if m.strip()]

    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 1234567890, "owned_by": "vllm"} for model_id in model_ids
        ],
    }


def generate_random_text(length=50):
    """Generate random but coherent text for responses."""
    words = [
        "Hello",
        "there",
        "I'm",
        "an",
        "AI",
        "assistant",
        "ready",
        "to",
        "help",
        "you",
        "with",
        "your",
        "questions",
        "and",
        "tasks",
        "today",
        "Let",
        "me",
        "know",
        "what",
        "you'd",
        "like",
        "to",
        "discuss",
        "or",
        "explore",
        "together",
        "I",
        "can",
        "assist",
        "with",
        "various",
        "topics",
        "including",
        "coding",
        "writing",
        "analysis",
        "and",
        "more",
    ]
    return " ".join(random.choices(words, k=length))


@app.route("/v1/models", methods=["GET"])
def list_models():
    models = get_models()
    print(f"[MOCK] Returning models: {[m['id'] for m in models['data']]}")
    return jsonify(models)


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Return OpenAI-formatted chat completion responses."""
    data = request.get_json()
    default_model = get_models()["data"][0]["id"]
    model = data.get("model", default_model)
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    print(f"[MOCK] Chat completion request - model: {model}, stream: {stream}")

    if stream:
        return handle_streaming_completion(model, messages)
    else:
        return handle_non_streaming_completion(model, messages)


def handle_non_streaming_completion(model, messages):
    response_text = generate_random_text(random.randint(20, 80))

    # Calculate realistic token counts
    prompt_tokens = sum(len(str(msg.get("content", "")).split()) for msg in messages)
    completion_tokens = len(response_text.split())

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    return jsonify(response)


def handle_streaming_completion(model, messages):
    def generate_stream():
        # Generate response text
        full_response = generate_random_text(random.randint(30, 100))
        words = full_response.split()

        # Send initial chunk
        initial_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        # Send word by word
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": f"{word} " if i < len(words) - 1 else word}}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Configurable delay to simulate realistic streaming
            stream_delay = float(os.getenv("STREAM_DELAY_SECONDS", "0.005"))
            time.sleep(stream_delay)

        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        generate_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "type": "openai-mock"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI-compatible mock server")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the server on (default: 8081)")
    args = parser.parse_args()

    port = args.port

    models = get_models()
    print("Starting OpenAI-compatible mock server...")
    print(f"- /models endpoint with: {[m['id'] for m in models['data']]}")
    print("- OpenAI-formatted chat/completion responses with dynamic content")
    print("- Streaming support with valid SSE format")
    print(f"- Listening on: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
