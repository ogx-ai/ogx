# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import pytest
from pydantic import BaseModel


class ResponsesTestCase(BaseModel):
    # Input can be a simple string or complex message structure
    input: str | list[dict[str, Any]]
    expected: str
    # Tools as flexible dict structure (gets validated at runtime by the API)
    tools: list[dict[str, Any]] | None = None
    # Multi-turn conversations with input/output pairs
    turns: list[tuple[str | list[dict[str, Any]], str]] | None = None
    # File search specific fields
    file_content: str | None = None
    file_path: str | None = None
    # Streaming flag
    stream: bool | None = None


# Basic response test cases
basic_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="Which planet do humans live on?",
            expected="earth",
        ),
        id="earth",
    ),
    pytest.param(
        ResponsesTestCase(
            input="Which planet has rings around it with a name starting with letter S?",
            expected="saturn",
        ),
        id="saturn",
    ),
    # TODO: Add image input test case, since this test case got Nerfed by OpenAI with a refusal
    # pytest.param(
    #     ResponsesTestCase(
    #         input=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "input_text",
    #                         "text": "what teams are playing in this image?",
    #                     }
    #                 ],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "input_image",
    #                         "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3b/LeBron_James_Layup_%28Cleveland_vs_Brooklyn_2018%29.jpg",
    #                     }
    #                 ],
    #             },
    #         ],
    #         expected="brooklyn nets",
    #     ),
    #     id="image_input",
    # ),
]

# Multi-turn test cases
multi_turn_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="",  # Not used for multi-turn
            expected="",  # Not used for multi-turn
            turns=[
                ("Which planet do humans live on?", "earth"),
                ("What is the name of the planet from your previous response?", "earth"),
            ],
        ),
        id="earth",
    ),
]

# Web search test cases
web_search_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="How many experts does the Llama 4 Maverick model have?",
            tools=[{"type": "web_search", "search_context_size": "low"}],
            expected="128",
        ),
        id="llama_experts",
    ),
    pytest.param(
        ResponsesTestCase(
            input="What is the latest version of Python?",
            tools=[{"type": "web_search_2025_08_26"}],
            expected="python",
        ),
        id="web_search_2025_08_26_type",
    ),
]

# File search test cases
file_search_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="How many experts does the Llama 4 Maverick model have?",
            tools=[{"type": "file_search"}],
            expected="128",
            file_content="Llama 4 Maverick has 128 experts",
        ),
        id="llama_experts",
    ),
    pytest.param(
        ResponsesTestCase(
            input="How many experts does the Llama 4 Maverick model have?",
            tools=[{"type": "file_search"}],
            expected="128",
            file_path="pdfs/ogx_and_models.pdf",
        ),
        id="llama_experts_pdf",
    ),
]

# MCP tool test cases
mcp_tool_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="What is the boiling point of myawesomeliquid in Celsius?",
            tools=[{"type": "mcp", "server_label": "localmcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}],
            expected="Hello, world!",
        ),
        id="boiling_point_tool",
    ),
]

# Custom tool test cases
custom_tool_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="What's the weather like in San Francisco?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {
                            "location": {
                                "description": "City and country e.g. Bogotá, Colombia",
                                "type": "string",
                            }
                        },
                        "required": ["location"],
                        "type": "object",
                    },
                }
            ],
            expected="",  # No specific expected output for custom tools
        ),
        id="sf_weather",
    ),
]

# Image test cases
image_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Identify the type of animal in this image.",
                        },
                        {
                            "type": "input_image",
                            "image_url": "https://upload.wikimedia.org/wikipedia/commons/f/f7/Llamas%2C_Vernagt-Stausee%2C_Italy.jpg",
                        },
                    ],
                },
            ],
            expected="llama",
        ),
        id="llama_image",
    ),
]

# Multi-turn image test cases
multi_turn_image_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="",  # Not used for multi-turn
            expected="",  # Not used for multi-turn
            turns=[
                (
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "What type of animal is in this image? Please respond with a single word that starts with the letter 'L'.",
                                },
                                {
                                    "type": "input_image",
                                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/f/f7/Llamas%2C_Vernagt-Stausee%2C_Italy.jpg",
                                },
                            ],
                        },
                    ],
                    "llama",
                ),
                (
                    "What country do you find this animal primarily in? What continent?",
                    "south america",
                ),
            ],
        ),
        id="llama_image_understanding",
    ),
]

# Multi-turn tool execution test cases
multi_turn_tool_execution_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="I need to check if user 'alice' can access the file 'document.txt'. First, get alice's user ID, then check if that user ID can access the file 'document.txt'. Do this as a series of steps, where each step is a separate message. Return only one tool call per step. Summarize the final result with a single 'yes' or 'no' response.",
            tools=[{"type": "mcp", "server_label": "localmcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}],
            expected="yes",
        ),
        id="user_file_access_check",
    ),
    pytest.param(
        ResponsesTestCase(
            input="I need to get the results for the 'boiling_point' experiment. First, get the experiment ID for 'boiling_point', then use that ID to get the experiment results. Tell me the boiling point in Celsius.",
            tools=[{"type": "mcp", "server_label": "localmcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}],
            expected="100°C",
        ),
        id="experiment_results_lookup",
    ),
]

# Multi-turn tool execution streaming test cases
multi_turn_tool_execution_streaming_test_cases = [
    pytest.param(
        ResponsesTestCase(
            input="Help me with this security check: First, get the user ID for 'charlie', then get the permissions for that user ID, and finally check if that user can access 'secret_file.txt'. Stream your progress as you work through each step. Return only one tool call per step. Summarize the final result with a single 'yes' or 'no' response.",
            tools=[{"type": "mcp", "server_label": "localmcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}],
            expected="no",
            stream=True,
        ),
        id="user_permissions_workflow",
    ),
    pytest.param(
        ResponsesTestCase(
            input="I need a complete analysis: First, get the experiment ID for 'chemical_reaction', then get the results for that experiment, and tell me if the yield was above 80%. Return only one tool call per step.  Please stream your analysis process.",
            tools=[{"type": "mcp", "server_label": "localmcp", "server_url": "<FILLED_BY_TEST_RUNNER>"}],
            expected="85%",
            stream=True,
        ),
        id="experiment_analysis_streaming",
    ),
]
