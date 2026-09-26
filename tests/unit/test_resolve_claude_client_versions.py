# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# The script lives outside the Python package tree, so import it by path.
import importlib.util
import json
import pathlib
from datetime import UTC, datetime

import pytest

_script_path = pathlib.Path(__file__).resolve().parents[2] / ".github" / "scripts" / "resolve_claude_client_versions.py"
_spec = importlib.util.spec_from_file_location("resolve_claude_client_versions", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _release(uploaded: str, yanked: bool = False) -> list[dict]:
    return [{"upload_time_iso_8601": uploaded, "yanked": yanked}]


PYPI = {
    "releases": {
        "0.2.99": _release("2026-08-01T00:00:00.000000Z"),
        "0.2.100": _release("2026-08-02T00:00:00.000000Z"),
        "0.2.157": _release("2026-09-18T18:21:15.969863Z"),  # 8 days old
        "0.2.158": _release("2026-09-23T01:39:18.829816Z"),
        "0.2.160": _release("2026-09-25T22:31:24.616384Z"),  # newest
        "0.3.0rc1": _release("2026-09-26T00:00:00.000000Z"),  # pre-release
        "0.2.161.dev1": _release("2026-09-26T01:00:00.000000Z"),
        "0.2.162": _release("2026-09-26T02:00:00.000000Z", yanked=True),
        "0.2.163": [],  # no files
    }
}


def _fetch(cli: dict[str, str] | None = None, pypi: dict | None = None):
    """A fake network: CLI channel endpoints plus the PyPI JSON document."""
    cli = cli or {"stable": "2.1.274", "latest": "2.1.283"}
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        if url == _mod.SDK_PYPI_URL:
            return json.dumps(pypi or PYPI)
        return next(text for channel, text in cli.items() if url.endswith(f"/{channel}"))

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


class TestResolveCliVersion:
    @pytest.mark.parametrize("channel,expected", [("stable", "2.1.274"), ("latest", "2.1.283")])
    def test_each_channel_resolves_from_its_own_endpoint(self, channel, expected):
        fetch = _fetch()

        assert _mod.resolve_cli_version(channel, fetch) == expected
        assert fetch.calls == [f"https://downloads.claude.ai/claude-code-releases/{channel}"]

    def test_stable_and_latest_differ(self):
        fetch = _fetch()

        assert _mod.resolve_cli_version("stable", fetch) != _mod.resolve_cli_version("latest", fetch)

    def test_surrounding_whitespace_is_ignored(self):
        assert _mod.resolve_cli_version("stable", _fetch(cli={"stable": " 2.1.274\n"})) == "2.1.274"

    def test_a_non_version_response_is_rejected(self):
        with pytest.raises(ValueError, match="Failed to parse the stable CLI version"):
            _mod.resolve_cli_version("stable", _fetch(cli={"stable": "<html>Service Unavailable</html>"}))

    def test_unknown_channel_is_rejected(self):
        with pytest.raises(ValueError, match="unknown channel 'beta'"):
            _mod.resolve_cli_version("beta", _fetch())


class TestResolveSdkVersion:
    def test_latest_is_the_newest_plain_release(self):
        assert _mod.resolve_sdk_version("latest", PYPI, NOW) == "0.2.160"

    def test_stable_is_the_newest_release_at_least_a_week_old(self):
        assert _mod.resolve_sdk_version("stable", PYPI, NOW) == "0.2.157"

    def test_a_release_exactly_at_the_cutoff_counts_as_stable(self):
        pypi = {"releases": {"1.0.0": _release("2026-09-20T12:00:00.000000Z")}}

        assert _mod.resolve_sdk_version("stable", pypi, NOW) == "1.0.0"

    def test_stable_moves_forward_as_releases_age(self):
        later = datetime(2026, 10, 3, 0, 0, tzinfo=UTC)

        assert _mod.resolve_sdk_version("stable", PYPI, later) == "0.2.160"

    def test_the_soak_period_is_configurable(self):
        assert _mod.resolve_sdk_version("stable", PYPI, NOW, stable_min_age_days=1) == "0.2.160"
        assert _mod.resolve_sdk_version("stable", PYPI, NOW, stable_min_age_days=30) == "0.2.100"

    def test_versions_are_ordered_numerically_not_lexically(self):
        pypi = {"releases": {"0.2.99": _release("2026-01-01T00:00:00Z"), "0.2.100": _release("2026-01-02T00:00:00Z")}}

        assert _mod.resolve_sdk_version("latest", pypi, NOW) == "0.2.100"

    @pytest.mark.parametrize("skipped", ["0.3.0rc1", "0.2.161.dev1", "0.2.162", "0.2.163"])
    def test_prereleases_yanked_and_empty_releases_are_never_chosen(self, skipped):
        assert _mod.resolve_sdk_version("latest", PYPI, NOW) != skipped

    def test_a_release_with_any_unyanked_file_is_usable(self):
        files = [
            {"upload_time_iso_8601": "2026-09-01T00:00:00Z", "yanked": True},
            {"upload_time_iso_8601": "2026-09-01T00:00:00Z", "yanked": False},
        ]

        assert _mod.resolve_sdk_version("latest", {"releases": {"1.2.3": files}}, NOW) == "1.2.3"

    def test_no_stable_release_yet_is_an_error(self):
        pypi = {"releases": {"0.1.0": _release("2026-09-26T00:00:00Z")}}

        with pytest.raises(RuntimeError, match="Failed to find a stable claude-agent-sdk release"):
            _mod.resolve_sdk_version("stable", pypi, NOW)


class TestResolve:
    def test_nightly_tests_latest_and_merge_ci_tests_stable(self):
        """The two channels must resolve to different clients, which is the point of having both."""
        stable = _mod.resolve("stable", fetch=_fetch(), now=NOW)
        latest = _mod.resolve("latest", fetch=_fetch(), now=NOW)

        assert stable == {"cli_version": "2.1.274", "sdk_version": "0.2.157"}
        assert latest == {"cli_version": "2.1.283", "sdk_version": "0.2.160"}

    def test_exact_overrides_win_over_the_channel_and_skip_the_lookup(self):
        fetch = _fetch()

        versions = _mod.resolve("latest", "2.1.100", "0.2.87", fetch=fetch)

        assert versions == {"cli_version": "2.1.100", "sdk_version": "0.2.87"}
        assert fetch.calls == []

    def test_each_override_applies_independently(self):
        versions = _mod.resolve("stable", cli_override="2.1.100", fetch=_fetch(), now=NOW)

        assert versions == {"cli_version": "2.1.100", "sdk_version": "0.2.157"}

    @pytest.mark.parametrize("bad", ["latest", "2.1", "2.1.1; rm -rf /", "1.2.3rc1", "$(id)"])
    def test_overrides_must_be_exact_release_versions(self, bad):
        with pytest.raises(ValueError, match="Failed to parse"):
            _mod.resolve("stable", cli_override=bad, fetch=_fetch())
        with pytest.raises(ValueError, match="Failed to parse"):
            _mod.resolve("stable", sdk_override=bad, fetch=_fetch())


class TestMain:
    def test_writes_github_output_and_prints_the_result(self, tmp_path, monkeypatch, capsys):
        output = tmp_path / "github_output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setattr(_mod, "_fetch", _fetch())
        monkeypatch.setattr(
            _mod, "resolve", lambda *args, **kwargs: {"cli_version": "2.1.283", "sdk_version": "0.2.160"}
        )

        assert _mod.main(["--channel", "latest"]) == 0

        assert output.read_text() == "cli_version=2.1.283\nsdk_version=0.2.160\n"
        assert "channel=latest" in capsys.readouterr().out

    def test_failure_is_reported_as_an_error_annotation(self, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        assert _mod.main(["--channel", "stable", "--cli-version", "nope"]) == 1

        assert "::error::Failed to parse the CLI version override" in capsys.readouterr().err
