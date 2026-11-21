# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

from openai import DefaultHttpxClient, OpenAI

client = OpenAI(
    base_url=os.environ["OGX_BASE_URL"],
    # Servers started with --no-auth accept any key
    api_key=os.environ.get("OGX_API_KEY", "unused"),
    # ogx go serves a self-signed certificate, so skip verification for local use
    http_client=DefaultHttpxClient(verify=False),
)

# SystemOne answers calibrated questions about a piece of text.
# The OpenAI SDK has no SystemOne method, so post to the endpoint directly.
judgment = client.post(
    "/systemone",
    cast_to=object,
    body={
        "model": "ollama/tev1:0.8b",
        "state": "I was charged twice for my order last week and nobody answers my emails.",
        "questions": {
            "is_complaint": {
                "type": "noul",
                "instructions": "Is the customer complaining?",
                "criteria": {"true": "The customer is unhappy about something", "false": "A neutral question or praise"},
            },
            "topic": {
                "type": "choice",
                "instructions": "What is the message about?",
                "criteria": {
                    "billing": "Payments, charges, refunds, or invoices",
                    "shipping": "Delivery, tracking, or lost packages",
                    "account": "Logging in, passwords, or profile settings",
                },
            },
        },
    },
)
print(json.dumps(judgment["answers"], indent=2))
