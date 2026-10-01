# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Stack-config-mode helpers shared by the integration suite and its unit tests.

Kept out of conftest.py so other modules can import them as a plain module instead of through
pytest's own conftest-loading machinery, which can give the same file two distinct module
objects depending on how it's reached -- silently breaking an `is`/set-membership check against
one of this module's constants if the checker imported a different copy than conftest.py did.
"""

from pathlib import Path


def is_server_stack_config(stack_config: str | None) -> bool:
    """Whether a --stack-config value points at a real server process (server:, docker:, or
    a bare http(s) URL) rather than an in-process library client.

    The single place this distinction is made from the --stack-config string; fixtures with
    access to `request` should prefer this over duplicating the prefix check, and module-level
    code that can't see the option at all should use pytest_collection_modifyitems in
    conftest.py instead of a hardcoded env var.
    """
    return stack_config is not None and stack_config.startswith(("server:", "docker:", "http"))


# Paths (relative to the repo root) whose gating needs config.getoption("--stack-config"),
# unavailable to a module-level `pytestmark = pytest.mark.skipif(...)`. Skipped by conftest.py's
# pytest_collection_modifyitems instead, once the config is available; see the modules
# themselves for why each is gated. Stored as Path.parts rather than a plain string so the
# match isn't sensitive to the OS path separator (str(Path(...)) renders backslashes on
# Windows).
SERVER_ONLY_TEST_PATHS = {Path("tests/integration/inspect/test_metrics_endpoint.py").parts}
LIBRARY_CLIENT_ONLY_TEST_PATHS = {Path("tests/integration/inference/test_inference_store_disabled.py").parts}
