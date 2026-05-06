# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re


def pascal_to_snake_case(name: str) -> str:
    """Convert PascalCase string to snake_case.

    Handles sequences like "OpenAI" -> "open_ai" correctly.

    :param name: PascalCase string
    :return: snake_case string
    :raises TypeError: If name is not a string
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected string, got {type(name).__name__}")

    # Handle sequences like "OpenAI" -> "Open_AI"
    snake = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Handle transitions like "Model1" -> "Model_1"
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake)
    return snake.lower()
