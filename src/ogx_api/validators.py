# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Validators for API request parameters.

This module contains validation functions used by providers to validate
request parameters that cannot be easily validated using Pydantic alone.
"""

from ogx_api.inference import OpenAIEmbeddingsRequestWithExtraBody


def validate_embeddings_input_is_text(
    params: OpenAIEmbeddingsRequestWithExtraBody,
) -> None:
    """
    Validate that embeddings input contains only text strings, not token arrays.

    Token arrays (list[int] and list[list[int]]) are a newer OpenAI feature
    that is not universally supported across all embedding providers. This
    validator should be called by providers that only support text input (str or list[str]).

    :param params: The OpenAI embeddings request parameters
    :raises ValueError: If input contains token arrays
    """
    # Valid: string input
    if isinstance(params.input, str):
        return

    # Valid: list of strings
    if isinstance(params.input, list) and isinstance(params.input[0], str):
        return

    # If we get here, input is a token array (list[int] or list[list[int]])
    raise ValueError(
        f"Model '{params.model}' does not support token arrays. "
        f"Please provide text input as a string or list of strings instead."
    )


__all__ = [
    "validate_embeddings_input_is_text",
]
