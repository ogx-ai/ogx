# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import requests
import urllib3

# ogx go serves a self-signed certificate, so skip verification for local use
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

response = requests.post(
    f"{os.environ['OGX_BASE_URL']}/systemone",
    headers={"Authorization": f"Bearer {os.environ.get('OGX_API_KEY', 'unused')}"},
    json={
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
                    "billing": "Payments, charges, or refunds, or invoices",
                    "shipping": "Delivery, tracking, or lost packages",
                    "account": "Logging in, passwords, or profile settings",
                },
            },
        },
    },
    verify=False,
)
response.raise_for_status()
print(json.dumps(response.json()["answers"], indent=2))
