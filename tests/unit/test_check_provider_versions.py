# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# The script lives outside the Python package tree, so import it by path.
import importlib.util
import pathlib
import textwrap
from typing import Any

import pytest

_script_path = pathlib.Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_provider_versions.py"
_spec = importlib.util.spec_from_file_location("check_provider_versions", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PROVIDERS = {p.key: p for p in _mod.PROVIDERS}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _release(tag: str, published_at: str, **extra: Any) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "published_at": published_at,
        "html_url": f"https://github.com/o/r/releases/tag/{tag}",
        **extra,
    }


class FakeApi:
    """Stands in for GitHubApi: canned upstream releases, and a tiny issue tracker."""

    def __init__(self, latest: dict[str, dict[str, Any]], llamacpp_pages: list[list[dict[str, Any]]] | None = None):
        self.latest = latest
        self.llamacpp_pages = llamacpp_pages or []
        self.issues: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    def request(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> Any:
        if method == "GET" and path.endswith("/releases/latest"):
            return self.latest[path.removeprefix("/repos/").removesuffix("/releases/latest")]
        if method == "GET" and path == "/repos/ggml-org/llama.cpp/releases":
            page = params["page"]
            return self.llamacpp_pages[page - 1] if page <= len(self.llamacpp_pages) else []
        if method == "GET" and path == "/search/issues":
            return {"items": list(self.issues)}
        if method == "POST" and path.endswith("/issues"):
            issue = {"title": body["title"], "html_url": f"https://github.com/o/r/issues/{len(self.issues) + 1}"}
            self.issues.append(issue)
            self.created.append({**issue, "body": body["body"]})
            return issue
        raise AssertionError(f"Unexpected API call: {method} {path}")


def _write(root: pathlib.Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


class TestVersions:
    def test_parse_version_strips_v_prefix(self):
        assert _mod.parse_version("v0.30.0").key == (0, 30, 0)
        assert _mod.parse_version("0.22.1").label == "0.22.1"

    def test_versions_compare_numerically_not_lexically(self):
        assert _mod.parse_version("0.30.0").key > _mod.parse_version("0.9.0").key
        assert _mod.parse_build("b10000").key > _mod.parse_build("b9999").key

    @pytest.mark.parametrize("text", ["0.30.0rc1", "latest", ""])
    def test_parse_version_rejects_non_release_tags(self, text):
        with pytest.raises(ValueError, match="Failed to parse"):
            _mod.parse_version(text)


class TestReadCurrentPin:
    def test_vllm_gpu_action_default(self, tmp_path):
        _write(
            tmp_path,
            ".github/actions/setup-vllm-gpu/action.yml",
            """\
            inputs:
              quantization:
                default: 'none'
              vllm-version:
                description: 'vLLM version to install'
                required: false
                default: '0.22.1'
              port:
                default: '8000'
            """,
        )
        assert _mod.read_current_pin(tmp_path, PROVIDERS["vllm"]).label == "0.22.1"

    def test_vllm_falls_back_to_native_cpu_wheel(self, tmp_path):
        _write(
            tmp_path,
            ".github/actions/setup-vllm/action.yml",
            """\
            run: |
              curl -o w "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cpu.whl"
            """,
        )
        assert _mod.read_current_pin(tmp_path, PROVIDERS["vllm"]).label == "0.22.1"

    def test_ollama_default(self, tmp_path):
        _write(
            tmp_path,
            ".github/actions/setup-ollama/action.yml",
            """\
            inputs:
              ollama-version:
                description: 'Ollama release'
                default: '0.34.2'
            """,
        )
        assert _mod.read_current_pin(tmp_path, PROVIDERS["ollama"]).key == (0, 34, 2)

    def test_llamacpp_build(self, tmp_path):
        _write(
            tmp_path,
            ".github/actions/setup-llamacpp/action.yml",
            '"https://github.com/ggml-org/llama.cpp/releases/download/b10964/llama-b10964-bin.tar.gz"\n',
        )
        pin = _mod.read_current_pin(tmp_path, PROVIDERS["llamacpp"])
        assert (pin.label, pin.key) == ("b10964", (10964,))

    def test_missing_pin_is_an_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to find the Ollama version pin"):
            _mod.read_current_pin(tmp_path, PROVIDERS["ollama"])

    def test_repository_pins_are_readable(self):
        """Guards against a setup action being reworked without updating this checker."""
        for provider in _mod.PROVIDERS:
            assert _mod.read_current_pin(REPO_ROOT, provider).key


class TestFindLocations:
    def test_matches_whole_versions_only(self, tmp_path):
        _write(
            tmp_path,
            "docs/gpu-runners.md",
            """\
            vLLM 0.22.1 is installed
            unrelated 10.22.1 and 0.22.10
            """,
        )
        provider = PROVIDERS["vllm"]
        locations = _mod.find_locations(tmp_path, provider, _mod.parse_version("0.22.1"))
        assert [(loc.path, loc.line) for loc in locations] == [("docs/gpu-runners.md", 1)]

    def test_includes_checksum_lines(self, tmp_path):
        _write(
            tmp_path,
            ".github/actions/setup-llamacpp/action.yml",
            """\
            LLAMA_TARBALL_SHA=abc
            url=releases/download/b10964/llama-b10964.tar.gz
            unrelated b109640
            """,
        )
        locations = _mod.find_locations(tmp_path, PROVIDERS["llamacpp"], _mod.parse_build("b10964"))
        assert [loc.line for loc in locations] == [1, 2]

    def test_skips_files_that_do_not_exist(self, tmp_path):
        assert _mod.find_locations(tmp_path, PROVIDERS["vllm"], _mod.parse_version("0.22.1")) == []


class TestSelectBlessedBuild:
    def test_latest_build_published_before_semver_release(self):
        """Real data: v0.4.1 (2026-09-14T18:27:29Z) maps to b10964."""
        releases = [
            _release("b10970", "2026-09-14T20:53:17Z"),
            _release("b10969", "2026-09-14T19:50:18Z"),
            _release("v0.4.1", "2026-09-14T18:27:29Z"),
            _release("b10964", "2026-09-14T15:28:07Z"),
            _release("b10956", "2026-09-14T13:57:29Z"),
        ]
        blessed = _mod.select_blessed_build(releases, "2026-09-14T18:27:29Z")
        assert blessed["tag_name"] == "b10964"

    def test_list_order_does_not_matter(self):
        releases = [_release("b1", "2026-01-01T00:00:00Z"), _release("b3", "2026-01-03T00:00:00Z")]
        assert _mod.select_blessed_build(releases, "2026-01-04T00:00:00Z")["tag_name"] == "b3"

    def test_build_published_at_the_same_instant_counts(self):
        releases = [_release("b5", "2026-01-01T00:00:00Z")]
        assert _mod.select_blessed_build(releases, "2026-01-01T00:00:00Z")["tag_name"] == "b5"

    def test_ignores_semver_tags_and_drafts(self):
        releases = [_release("v0.3.0", "2026-01-01T00:00:00Z"), _release("b9", "2026-01-01T00:00:00Z", draft=True)]
        assert _mod.select_blessed_build(releases, "2026-02-01T00:00:00Z") is None


class TestResolveLatestLlamacpp:
    def test_pages_until_a_blessed_build_is_found_and_reads_checksum(self):
        tarball = "llama-b11147-bin-ubuntu-x64.tar.gz"
        api = FakeApi(
            latest={"ggml-org/llama.cpp": _release("v0.5.0", "2026-09-23T20:50:06Z")},
            llamacpp_pages=[
                [_release("b11149", "2026-09-23T20:56:21Z")],
                [
                    _release(
                        "b11147",
                        "2026-09-23T19:03:52Z",
                        assets=[{"name": tarball, "digest": "sha256:" + "ab" * 32}],
                    )
                ],
            ],
        )
        latest = _mod.resolve_latest_llamacpp(api, PROVIDERS["llamacpp"])
        assert latest.version.label == "b11147"
        assert latest.release_tag == "v0.5.0"
        assert latest.tarball_name == tarball
        assert latest.tarball_sha256 == "ab" * 32

    def test_no_blessed_build_is_an_error(self):
        api = FakeApi(latest={"ggml-org/llama.cpp": _release("v0.5.0", "2026-09-23T20:50:06Z")})
        with pytest.raises(RuntimeError, match="Failed to find a llama.cpp build"):
            _mod.resolve_latest_llamacpp(api, PROVIDERS["llamacpp"])


class TestIssueRendering:
    def test_body_lists_locations_and_backticks_do_not_break_code_spans(self):
        result = _mod.Result(provider=PROVIDERS["vllm"])
        result.current = _mod.parse_version("0.22.1")
        result.latest = _mod.Latest(_mod.parse_version("0.30.0"), "https://example/v0.30.0", "2026-09-22T05:20:54Z")
        result.locations = [_mod.Location("docs/x.md", 7, "uses `/tmp/env` from 0.22.1")]
        body = _mod.render_issue_body(result, REPO_ROOT, "https://example/run")
        assert "`docs/x.md:7`" in body
        assert "`` uses `/tmp/env` from 0.22.1 ``" in body
        assert "[v0.30.0](https://example/v0.30.0)" in body
        assert _mod.issue_title(result) == "Update vLLM test server pin to v0.30.0"


class TestMain:
    """End to end against the repository's real pins with a fake GitHub API."""

    @pytest.fixture
    def api(self, monkeypatch):
        fake = FakeApi(
            latest={
                "vllm-project/vllm": _release("v99.0.0", "2026-09-22T05:20:54Z"),
                "ollama/ollama": _release("v99.0.0", "2026-09-23T02:24:43Z"),
                "ggml-org/llama.cpp": _release("v99.0.0", "2026-09-23T20:50:06Z"),
            },
            llamacpp_pages=[[_release("b99999", "2026-09-23T19:03:52Z")]],
        )
        monkeypatch.setattr(_mod, "GitHubApi", lambda token: fake)
        return fake

    def _run(self, *extra: str) -> int:
        return _mod.main(["--repo-root", str(REPO_ROOT), "--target-repo", "o/r", *extra])

    def test_files_one_issue_per_outdated_provider_and_never_duplicates(self, api, capsys):
        assert self._run() == 0
        assert [i["title"] for i in api.created] == [
            "Update vLLM test server pin to v99.0.0",
            "Update Ollama test server pin to v99.0.0",
            "Update llama.cpp test server pin to b99999 (v99.0.0)",
        ]
        summary = capsys.readouterr().out
        for name in ("vLLM", "Ollama", "llama.cpp"):
            assert f"| {name} |" in summary

        assert self._run() == 0
        assert len(api.created) == 3
        assert "exists" in capsys.readouterr().out

    def test_dry_run_creates_nothing(self, api, capsys):
        assert self._run("--dry-run") == 0
        assert api.created == []
        assert "would create" in capsys.readouterr().out

    def test_up_to_date_provider_files_no_issue(self, api, capsys):
        api.latest["ollama/ollama"] = _release("v0.0.1", "2026-09-23T02:24:43Z")
        assert self._run() == 0
        assert not any("Ollama" in i["title"] for i in api.created)
        assert "up to date" in capsys.readouterr().out

    def test_provider_error_fails_the_run_but_still_reports_the_others(self, api, capsys):
        api.latest["ollama/ollama"] = _release("nightly", "2026-09-23T02:24:43Z")
        assert self._run() == 1
        assert len(api.created) == 2
        assert "Failed to parse version" in capsys.readouterr().out
