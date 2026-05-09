#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# /// script
# dependencies = [
#   "google-genai",
# ]
# ///

"""Capture Google Interactions API responses as test fixtures.

Calls the real Google Interactions API and saves the raw responses as JSON
fixtures. These fixtures are used by unit tests to validate that OGX's
Interactions API translation layer produces responses matching Google's format.

Re-run this script periodically to catch API changes.

Usage:
    # With Gemini API key:
    GEMINI_API_KEY=<key> uv run scripts/capture_google_interactions_fixtures.py

    # With Vertex AI:
    uv run scripts/capture_google_interactions_fixtures.py --vertex --project <project> --location us-central1

    # With a different model:
    GEMINI_API_KEY=<key> uv run scripts/capture_google_interactions_fixtures.py --model gemini-2.5-flash
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "unit" / "providers" / "inline" / "interactions" / "fixtures"


def _sanitize(obj):
    """Convert non-serializable types to strings."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _sanitize(obj.to_dict())
    return str(obj) if not isinstance(obj, str | int | float | bool | None) else obj


def capture_fixtures(client, model: str) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Non-streaming
    print(f"Capturing non-streaming response from {model}...")
    r = client.interactions.create(
        model=model,
        input="What is 2+2? Reply with just the number.",
    )
    fixture = _sanitize(r.to_dict())
    path = FIXTURES_DIR / "google_non_streaming.json"
    path.write_text(json.dumps(fixture, indent=2, default=str) + "\n")
    print(f"  Saved: {path}")

    # Streaming
    print(f"Capturing streaming response from {model}...")
    stream = client.interactions.create(
        model=model,
        input="Say hi in one word.",
        stream=True,
    )
    events = []
    for event in stream:
        events.append({"event_type": type(event).__name__, "data": _sanitize(event.to_dict())})
    path = FIXTURES_DIR / "google_streaming.json"
    path.write_text(json.dumps(events, indent=2, default=str) + "\n")
    print(f"  Saved: {path}")

    print("Done.")


def main():
    from google import genai

    parser = argparse.ArgumentParser(description="Capture Google Interactions API fixtures")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use (default: gemini-2.5-flash)")
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of Gemini API key")
    parser.add_argument("--project", default=None, help="GCP project for Vertex AI")
    parser.add_argument("--location", default="us-central1", help="GCP location for Vertex AI")
    args = parser.parse_args()

    if args.vertex:
        if not args.project:
            print("ERROR: --project required with --vertex", file=sys.stderr)
            sys.exit(1)
        client = genai.Client(vertexai=True, project=args.project, location=args.location)
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey", file=sys.stderr)
            sys.exit(1)
        client = genai.Client(api_key=api_key)

    capture_fixtures(client, args.model)


if __name__ == "__main__":
    main()
