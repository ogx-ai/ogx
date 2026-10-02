# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Exercise release workflow shell decisions without publishing artifacts."""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _step(workflow: str, job: str, step_id: str) -> dict[str, Any]:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return next(step for step in document["jobs"][job]["steps"] if step.get("id", step.get("name")) == step_id)


def _run(
    step: dict[str, Any], tmp_path: Path, values: dict[str, str], prefix: str = ""
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        return values.get(expression, "")

    output = tmp_path / "output"
    output.write_text("")
    env = {**os.environ, "GITHUB_OUTPUT": str(output), "GITHUB_REPOSITORY": "ogx-ai/ogx"}
    env.update({key: re.sub(r"\$\{\{(.*?)\}\}", replace, str(value)) for key, value in step.get("env", {}).items()})
    script = re.sub(r"\$\{\{(.*?)\}\}", replace, step["run"])
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", prefix + "\n" + script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    return result, outputs


@pytest.mark.parametrize("step_id", ["version-check", "npm-version-check"])
@pytest.mark.parametrize("version,expected", [("1.0.3", "1.0.3"), ("1.3.2.dev20260909", "1.3.2.dev20260909")])
def test_client_build_keeps_requested_version_when_registry_has_it(
    tmp_path: Path, step_id: str, version: str, expected: str
) -> None:
    result, outputs = _run(
        _step("pypi.yml", "build-package", step_id),
        tmp_path,
        {"needs.compute-version.outputs.version": version, "matrix.package": "ogx-client-python"},
        "curl() { return 0; }",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["version"] == (expected.replace(".dev", "-dev.") if step_id == "npm-version-check" else expected)


@pytest.mark.parametrize("job,step_id", [("build-package", "should-build"), ("publish-packages", "should-publish")])
@pytest.mark.parametrize(
    "event,selection,kind,skip",
    [
        ("release", "all", "external", "true"),
        ("release", "all", "openapi-sdk", "true"),
        ("release", "all", "local", "false"),
        ("workflow_dispatch", "ogx-only", "openapi-sdk", "true"),
        ("workflow_dispatch", "ogx-only", "local", "false"),
        ("workflow_dispatch", "clients-only", "openapi-sdk", "false"),
        ("workflow_dispatch", "clients-only", "external", "false"),
        ("workflow_dispatch", "clients-only", "local", "true"),
        ("push", "all", "external", "false"),
    ],
)
def test_package_selection(
    tmp_path: Path, job: str, step_id: str, event: str, selection: str, kind: str, skip: str
) -> None:
    result, outputs = _run(
        _step("pypi.yml", job, step_id),
        tmp_path,
        {
            "github.event_name": event,
            "inputs.packages || 'all'": selection,
            "matrix.type": kind,
            "matrix.package": "test-package",
        },
    )
    assert result.returncode == 0, result.stderr
    assert outputs["skip"] == skip


@pytest.mark.parametrize(
    "event,version,latest,skip,latest_tag",
    [
        ("release", "1.0.3", "v1.3.1", "false", False),
        ("release", "1.3.1", "v1.3.1", "false", True),
        ("release", "1.10.0", "v1.9.9", "false", True),
        ("release", "1.3.2rc1", "v1.3.1", "false", False),
        ("workflow_dispatch", "1.0.3", "v1.3.1", "true", False),
        ("workflow_dispatch", "1.3.2", "v1.3.1", "false", True),
    ],
)
def test_docker_latest_tag(tmp_path: Path, event: str, version: str, latest: str, skip: str, latest_tag: bool) -> None:
    result, outputs = _run(
        _step("pypi.yml", "publish-docker-images", "meta"),
        tmp_path,
        {
            "github.event_name": event,
            "inputs.dry_run": "off",
            "inputs.skip_latest": skip,
            "inputs.package_name || 'ogx'": "ogx",
            "matrix.distro": "starter",
            "needs.compute-version.outputs.version": version,
        },
        f"gh() {{ printf '%s\\n' '{latest}'; }}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["tags"] == "ogxai/distribution-starter:" + version + (
        ",ogxai/distribution-starter:latest" if latest_tag else ""
    )


@pytest.mark.parametrize("command", ["return 1", "printf '%s\\n' invalid"])
def test_docker_fails_closed_when_latest_release_is_unavailable(tmp_path: Path, command: str) -> None:
    result, outputs = _run(
        _step("pypi.yml", "publish-docker-images", "meta"),
        tmp_path,
        {
            "github.event_name": "release",
            "inputs.skip_latest": "false",
            "matrix.distro": "starter",
            "inputs.package_name || 'ogx'": "ogx",
            "needs.compute-version.outputs.version": "1.0.3",
        },
        f"gh() {{ {command}; }}",
    )
    assert result.returncode != 0
    assert "tags" not in outputs


@pytest.mark.parametrize("skip,expected", [("true", "release-1.0"), ("false", "latest")])
def test_npm_backfill_uses_release_stream_dist_tag(tmp_path: Path, skip: str, expected: str) -> None:
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "client.tgz").touch()
    result, _ = _run(
        _step("pypi.yml", "publish-packages", "Publish to npm"),
        tmp_path,
        {
            "github.event_name": "workflow_dispatch",
            "inputs.dry_run": "off",
            "inputs.skip_latest": skip,
            "needs.compute-version.outputs.version": "1.0.3",
            "secrets.NPM_TOKEN": "test-token",
        },
        "npm() { printf '%s\\n' \"$@\" > npm-arguments; }",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = (tmp_path / "npm-arguments").read_text().splitlines()
    assert arguments[arguments.index("--tag") + 1] == expected


@pytest.mark.parametrize(
    "version,current,update",
    [
        ("1.0.3", "1.3.1.dev0", "false"),
        ("1.3.1", "1.3.1.dev0", "true"),
        ("1.10.0", "1.9.9.dev0", "true"),
        ("1.3.2rc1", "1.3.1.dev0", "false"),
        ("1.0.3rc1", "1.3.1.dev0", "false"),
    ],
)
def test_old_release_does_not_change_main_version(tmp_path: Path, version: str, current: str, update: str) -> None:
    result, outputs = _run(
        _step("post-release.yml", "post-release", "main-version"),
        tmp_path,
        {
            "steps.parse.outputs.version": version,
        },
        f"git() {{ printf '%s\\n' 'fallback_version = \"{current}\"'; }}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["update_main"] == update


def _package_readiness_step() -> dict[str, Any]:
    document = yaml.safe_load((WORKFLOWS / "pypi.yml").read_text())
    return next(
        step
        for step in document["jobs"]["publish-docker-images"]["steps"]
        if step["name"].startswith("Wait for package")
    )


@pytest.mark.parametrize("mode,host", [("pypi", "pypi.org"), ("test-pypi", "test.pypi.org")])
def test_docker_waits_for_both_packages_on_the_selected_index(tmp_path: Path, mode: str, host: str) -> None:
    step = _package_readiness_step()
    assert not step.get("if"), "The readiness check must run for production as well as TestPyPI"
    result, _ = _run(
        step,
        tmp_path,
        {
            "needs.compute-version.outputs.version": "1.0.3",
            "steps.meta.outputs.install_mode": mode,
            "steps.meta.outputs.package_name || 'ogx'": "ogx",
        },
        r"""
        curl() {
          local argument url
          for argument in "$@"; do
            case "$argument" in https://*) url="$argument" ;; esac
          done
          printf '%s\n' "$url" >> requests
          case "$url" in
            */ogx-api/)
              if [ ! -f api-requested ]; then touch api-requested; return 22; fi
              printf '%s\n' '{"meta":{"api-version":"1.0"},"files":[{"filename":"ogx_api-1.0.3-py3-none-any.whl","yanked":false}]}' ;;
            */ogx/)
              printf '%s\n' '{"meta":{"api-version":"1.0"},"files":[{"filename":"ogx-1.0.3-py3-none-any.whl","yanked":false}]}' ;;
            *) return 22 ;;
          esac
        }
        sleep() { printf '%s\n' "$1" >> sleeps; }
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    requests = (tmp_path / "requests").read_text().splitlines()
    assert requests.count(f"https://{host}/simple/ogx-api/") == 2
    assert requests.count(f"https://{host}/simple/ogx/") == 2
    assert (tmp_path / "sleeps").read_text().splitlines() == ["30"]


@pytest.mark.parametrize(
    "unavailable_file",
    [
        '{"filename":"ogx-1.0.2-py3-none-any.whl","yanked":false}',
        '{"filename":"ogx-1.0.3.tar.gz","yanked":false}',
        '{"filename":"ogx-1.0.3-py3-none-any.whl","yanked":"withdrawn"}',
        '{"filename":"ogx-1.0.3-py3-none-any.whl","yanked":""}',
        '{"filename":"ogx-1.0.3-py3-none-any.whl","yanked":true}',
    ],
)
def test_docker_stops_after_bounded_wait_for_missing_or_yanked_release(tmp_path: Path, unavailable_file: str) -> None:
    (tmp_path / "index.json").write_text('{"meta":{"api-version":"1.0"},"files":[' + unavailable_file + "]}")
    result, _ = _run(
        _package_readiness_step(),
        tmp_path,
        {
            "needs.compute-version.outputs.version": "1.0.3",
            "steps.meta.outputs.install_mode": "pypi",
            "steps.meta.outputs.package_name || 'ogx'": "ogx",
        },
        'curl() { echo request >> requests; cat index.json; }\nsleep() { echo "$1" >> sleeps; }',
    )
    assert result.returncode != 0
    assert len((tmp_path / "requests").read_text().splitlines()) == 40
    assert (tmp_path / "sleeps").read_text().splitlines() == ["30"] * 19


def test_docker_wait_preserves_legacy_releases_with_bundled_api(tmp_path: Path) -> None:
    result, _ = _run(
        _package_readiness_step(),
        tmp_path,
        {
            "needs.compute-version.outputs.version": "0.5.0",
            "steps.meta.outputs.install_mode": "pypi",
            "steps.meta.outputs.package_name || 'ogx'": "llama-stack",
        },
        r"""
        curl() {
          printf '%s\n' "$@" >> requests
          printf '%s\n' '{"meta":{"api-version":"1.0"},"files":[{"filename":"llama_stack-0.5.0-py3-none-any.whl","yanked":false}]}'
        }
        sleep() { return 1; }
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    requests = (tmp_path / "requests").read_text()
    assert "https://pypi.org/simple/llama-stack/" in requests
    assert "llama-stack-api" not in requests
