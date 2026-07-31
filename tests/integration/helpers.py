# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unicodedata


def normalize_text(text: str) -> str:
    """Normalize Unicode text for comparison by stripping diacritical marks and non-ASCII characters.

    Models often return typographically correct but comparison-hostile text:
    narrow no-break spaces between numbers and units (``100\\u202f°C``),
    macrons on Latin words (``sōl``), etc.  NFD-decomposing then encoding
    to ASCII strips all such variation so that simple ``in`` checks work.
    """
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def assert_text_contains(text: str, expected: str, msg: str | None = None):
    """Assert that *expected* appears in *text* after Unicode normalisation."""
    normalized = normalize_text(text)
    normalized_expected = normalize_text(expected)
    assert normalized_expected in normalized, msg or f"Expected '{expected}' in text: {text}"
