# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("install_exit_code", [0, 42])
def test_integration_runner_stops_after_failed_dependency_install(tmp_path: Path, install_exit_code: int) -> None:
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    install_log = tmp_path / "install.log"
    pytest_log = tmp_path / "pytest.log"
    commands = {
        "uv": """#!/bin/bash
case "$1 $2" in
    "pip list")
        if [[ "${3:-}" != "--format=freeze" ]]; then
            echo "ogx 1.0.2"
        fi
        ;;
    "pip install")
        echo "$*" > "$INSTALL_LOG"
        exit "$INSTALL_EXIT_CODE"
        ;;
    *) exit 99 ;;
esac
""",
        "ogx": """#!/bin/bash
if [[ "$1 $2" == "stack list-deps" ]]; then
    echo "sentence-transformers>=2"
else
    exit 99
fi
""",
        "python": "#!/bin/bash\nexit 0\n",
        "pytest": '#!/bin/bash\necho "$*" > "$PYTEST_LOG"\n',
    }
    for name, contents in commands.items():
        command = command_dir / name
        command.write_text(contents)
        command.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{command_dir}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "INSTALL_EXIT_CODE": str(install_exit_code),
        "INSTALL_LOG": str(install_log),
        "PYTEST_LOG": str(pytest_log),
    }
    environment.pop("TS_CLIENT_PATH", None)
    environment.pop("INTEGRATION_TESTS_POST_CMD", None)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/integration-tests.sh"),
            "--stack-config",
            "ci-tests",
            "--setup",
            "gpt",
            "--install-deps",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert install_log.read_text().strip() == "pip install sentence-transformers>=2"
    assert result.returncode == (0 if install_exit_code == 0 else 1), result.stdout + result.stderr
    assert pytest_log.exists() == (install_exit_code == 0)


def _sdk_pyproject(root: Path) -> tuple[Path, str]:
    """Return (pyproject, version) for the generated Python SDK, creating a
    fixture if the gitignored SDK has not been generated in this checkout."""
    pyproject = root / "client-sdks" / "openapi" / "sdks" / "python" / "pyproject.toml"
    if not pyproject.exists():
        pyproject.parent.mkdir(parents=True, exist_ok=True)
        pyproject.write_text('version = "9.9.9"\n')
    version = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert version is not None
    return pyproject, version.group(1)


def _run_runner(tmp_path: Path, root: Path, environment: dict, *args: str) -> "subprocess.Completed[str]":
    return subprocess.run(
        ["bash", str(root / "scripts/integration-tests.sh"), *args],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("version_matches", [True, False])
def test_integration_runner_client_version_latest(tmp_path: Path, version_matches: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    _, expected_version = _sdk_pyproject(root)
    install_log = tmp_path / "install.log"
    make_log = tmp_path / "make.log"
    pytest_log = tmp_path / "pytest.log"
    installed_version = expected_version if version_matches else "0.0.0"
    commands = {
        "uv": """#!/bin/bash
case "$1 $2" in
    "pip show")
        case "$3" in
            ogx-client) echo "Version: $CLIENT_VERSION_OUT" ;;
            ogx|ogx-api) echo "Editable project location: /tmp" ;;
            *) exit 1 ;;
        esac
        ;;
    "pip install")
        echo "$*" >> "$INSTALL_LOG"
        ;;
    "pip list")
        if [[ "${3:-}" == "--format=freeze" ]]; then
            echo "sentence-transformers=2.0"
        fi
        echo "ogx-client $CLIENT_VERSION_OUT"
        ;;
    *) exit 99 ;;
esac
""",
        "make": """#!/bin/bash
echo "make $*" >> "$MAKE_LOG"
""",
        "java": "#!/bin/bash\nexit 0\n",
        "node": "#!/bin/bash\nexit 0\n",
        "openapi-generator-cli": "#!/bin/bash\nexit 0\n",
        "ogx": """#!/bin/bash
if [[ "$1 $2" == "stack list-deps" ]]; then
    echo "sentence-transformers>=2"
else
    exit 99
fi
""",
        "python": "#!/bin/bash\nexit 0\n",
        "pytest": '#!/bin/bash\necho "$*" > "$PYTEST_LOG"\n',
    }
    for name, contents in commands.items():
        command = tmp_path / name
        command.write_text(contents)
        command.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "CLIENT_VERSION_OUT": installed_version,
        "INSTALL_LOG": str(install_log),
        "MAKE_LOG": str(make_log),
        "PYTEST_LOG": str(pytest_log),
    }
    environment.pop("TS_CLIENT_PATH", None)
    environment.pop("INTEGRATION_TESTS_POST_CMD", None)
    result = _run_runner(
        tmp_path,
        root,
        environment,
        "--stack-config",
        "ci-tests",
        "--setup",
        "gpt",
        "--client-version",
        "latest",
    )

    assert "sdk OPEN=0" in make_log.read_text()
    install_args = install_log.read_text().strip()
    assert install_args.startswith("pip install --upgrade ")
    assert install_args.endswith("client-sdks/openapi/sdks/python")
    assert result.returncode == (0 if version_matches else 1), result.stdout + result.stderr
    assert pytest_log.exists() == version_matches


def test_integration_runner_client_version_published(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    lock_text = (root / "uv.lock").read_text()
    match = re.search(r'^name = "ogx-client"\nversion = "([^"]+)"', lock_text, re.MULTILINE)
    assert match is not None
    pytest_log = tmp_path / "pytest.log"
    commands = {
        "uv": """#!/bin/bash
case "$1 $2" in
    "pip show")
        case "$3" in
            ogx-client) echo "Version: $CLIENT_VERSION_OUT" ;;
            ogx|ogx-api) echo "Editable project location: /tmp" ;;
            *) exit 1 ;;
        esac
        ;;
    "pip list")
        if [[ "${3:-}" == "--format=freeze" ]]; then
            echo "sentence-transformers=2.0"
        fi
        echo "ogx-client $CLIENT_VERSION_OUT"
        ;;
    *) exit 99 ;;
esac
""",
        "ogx": """#!/bin/bash
if [[ "$1 $2" == "stack list-deps" ]]; then
    echo "sentence-transformers>=2"
else
    exit 99
fi
""",
        "python": "#!/bin/bash\nexit 0\n",
        "pytest": '#!/bin/bash\necho "$*" > "$PYTEST_LOG"\n',
    }
    for name, contents in commands.items():
        command = tmp_path / name
        command.write_text(contents)
        command.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "CLIENT_VERSION_OUT": match.group(1),
        "PYTEST_LOG": str(pytest_log),
    }
    environment.pop("TS_CLIENT_PATH", None)
    environment.pop("INTEGRATION_TESTS_POST_CMD", None)
    result = _run_runner(
        tmp_path,
        root,
        environment,
        "--stack-config",
        "ci-tests",
        "--setup",
        "gpt",
        "--client-version",
        "published",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert pytest_log.exists()


def test_integration_runner_client_version_invalid(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    pytest_log = tmp_path / "pytest.log"
    commands = {
        "uv": """#!/bin/bash
case "$1 $2" in
    "pip list")
        if [[ "${3:-}" == "--format=freeze" ]]; then
            echo "sentence-transformers=2.0"
        fi
        echo "ogx-client 1.0.0"
        ;;
    *) exit 99 ;;
esac
""",
        "ogx": """#!/bin/bash
if [[ "$1 $2" == "stack list-deps" ]]; then
    echo "sentence-transformers>=2"
else
    exit 99
fi
""",
        "python": "#!/bin/bash\nexit 0\n",
        "pytest": '#!/bin/bash\necho "$*" > "$PYTEST_LOG"\n',
    }
    for name, contents in commands.items():
        command = tmp_path / name
        command.write_text(contents)
        command.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "PYTEST_LOG": str(pytest_log),
    }
    environment.pop("TS_CLIENT_PATH", None)
    environment.pop("INTEGRATION_TESTS_POST_CMD", None)
    result = _run_runner(
        tmp_path,
        root,
        environment,
        "--stack-config",
        "ci-tests",
        "--setup",
        "gpt",
        "--client-version",
        "bogus",
    )

    assert result.returncode == 1
    assert "Unknown client-version" in result.stderr
    assert not pytest_log.exists()
