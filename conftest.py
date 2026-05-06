# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Root-level conftest.py for pytest plugin loading.

In pytest 8.4+, pytest_plugins must be defined at the rootdir conftest.py level,
not in subdirectory conftest files. This file dynamically loads the appropriate
fixture plugins based on which tests are being collected.
"""

from pathlib import Path


def pytest_configure(config):
    """Dynamically import fixture plugins based on test collection paths."""
    args = config.invocation_params.args
    rootpath = Path(str(config.rootpath))

    def _looks_like_existing_test_path(arg: object) -> bool:
        """Return True if arg is an explicit test path / nodeid.

        Pytest invocations can include option values like `-k foo`, where "foo" is
        not a test path. Pytest also accepts node ids like
        `tests/unit/test_x.py::test_name`.
        """

        if arg is None:
            return False

        arg_str = str(arg)
        if not arg_str or arg_str.startswith("-"):
            return False

        if arg_str in (".", "tests", "tests/"):
            return True

        path_part = arg_str.split("::", 1)[0]
        if not path_part:
            return False

        p = Path(path_part)
        if p.is_absolute():
            return p.exists()

        return (rootpath / p).exists()

    # Check if we're running unit tests
    running_unit = any("tests/unit" in str(arg) or "tests\\unit" in str(arg) for arg in args)
    # Check if we're running integration tests
    running_integration = any("tests/integration" in str(arg) or "tests\\integration" in str(arg) for arg in args)

    # If no test path explicitly given, load both plugin sets.
    # Only count an arg as a "path" if it actually exists as a file/dir.
    # This avoids misclassifying option values like "-k foo" where "foo" is not a path.
    has_explicit_path_args = any(_looks_like_existing_test_path(arg) for arg in args)

    # If user didn't pass a path, pytest will collect from the root by default.
    # This covers: pytest, pytest -v, pytest -k foo, etc.
    if not has_explicit_path_args and not running_unit and not running_integration:
        running_unit = True
        running_integration = True

    # Import plugins dynamically
    if running_unit:
        config.pluginmanager.import_plugin("tests.unit.fixtures")

    if running_integration:
        config.pluginmanager.import_plugin("tests.integration.fixtures.common")
