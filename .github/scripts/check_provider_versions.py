#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Detect stale provider-server pins used for integration test recording.

Recordings are produced against live vLLM, Ollama and llama.cpp servers whose
versions are pinned in the CI setup actions. For each provider this script reads
the current pin, asks the upstream GitHub repository for its latest release and,
when the pin is behind, opens an issue (deduplicated by title, which embeds the
new version) listing every file and line that must change.

llama.cpp publishes a semver release (vX.Y.Z) plus a stream of bNNNN build
prereleases, and our setup action pins a build. The "blessed" build for a
semver release is the latest bNNNN prerelease published on or before it.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
LLAMACPP_ASSET_TEMPLATE = "llama-{tag}-bin-ubuntu-x64.tar.gz"
LLAMACPP_MAX_PAGES = 30

VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)*)$")
BUILD_TAG_RE = re.compile(r"^b(\d+)$")


@dataclass(frozen=True)
class Version:
    """A release version: `label` for display, `key` for numeric comparison."""

    label: str
    key: tuple[int, ...]


def parse_version(text: str) -> Version:
    match = VERSION_RE.match(text.strip())
    if match is None:
        raise ValueError(f"Failed to parse version from {text!r}")
    return Version(label=match.group(1), key=tuple(int(part) for part in match.group(1).split(".")))


def parse_build(text: str) -> Version:
    match = BUILD_TAG_RE.match(text.strip())
    if match is None:
        raise ValueError(f"Failed to parse llama.cpp build tag from {text!r}")
    return Version(label=f"b{match.group(1)}", key=(int(match.group(1)),))


def _input_default_pattern(input_name: str) -> str:
    """Regex capturing the default of a composite-action input, e.g. `vllm-version`."""
    return rf"^[ \t]*{re.escape(input_name)}:[ \t]*\n(?:[ \t]+[^\n]*\n)*?[ \t]+default:[ \t]*'?(\d+(?:\.\d+)+)'?"


@dataclass(frozen=True)
class PinSource:
    """A file and regex (first group = version) that defines a provider pin."""

    path: str
    pattern: str


@dataclass(frozen=True)
class Provider:
    """A provider server whose pinned version is tracked against upstream releases."""

    key: str
    display: str
    upstream: str
    pin_sources: tuple[PinSource, ...]
    # Files that mention the pinned version; every matching line is listed in the issue.
    location_files: tuple[str, ...]
    # Extra lines (e.g. checksum assignments) that must change alongside the version.
    extra_location_pattern: str | None = None


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        key="vllm",
        display="vLLM",
        upstream="vllm-project/vllm",
        pin_sources=(
            PinSource(".github/actions/setup-vllm-gpu/action.yml", _input_default_pattern("vllm-version")),
            PinSource(".github/actions/setup-vllm/action.yml", r"releases/download/v(\d+(?:\.\d+)+)/vllm-"),
        ),
        location_files=(
            ".github/actions/setup-vllm-gpu/action.yml",
            ".github/actions/setup-vllm/action.yml",
            ".github/workflows/launch-gpu-ec2-runner.yml",
            ".github/workflows/record-vllm-gpu-tests.yml",
            "docs/gpu-runners.md",
        ),
        extra_location_pattern=r"WHEEL_SHA=",
    ),
    Provider(
        key="ollama",
        display="Ollama",
        upstream="ollama/ollama",
        pin_sources=(PinSource(".github/actions/setup-ollama/action.yml", _input_default_pattern("ollama-version")),),
        location_files=(
            ".github/actions/setup-ollama/action.yml",
            "tests/containers/ollama-with-models.containerfile",
            "tests/containers/ollama-with-vision-model.containerfile",
        ),
    ),
    Provider(
        key="llamacpp",
        display="llama.cpp",
        upstream="ggml-org/llama.cpp",
        pin_sources=(PinSource(".github/actions/setup-llamacpp/action.yml", r"releases/download/(b\d+)/"),),
        location_files=(".github/actions/setup-llamacpp/action.yml",),
        extra_location_pattern=r"LLAMA_TARBALL_SHA=",
    ),
)


@dataclass(frozen=True)
class Location:
    """A line in the repository that mentions a pinned version."""

    path: str
    line: int
    text: str


@dataclass
class Latest:
    """The latest upstream release for a provider."""

    version: Version
    url: str
    published_at: str
    # llama.cpp only: the semver release the blessed build belongs to, and the tarball checksum.
    release_tag: str = ""
    tarball_name: str = ""
    tarball_sha256: str = ""


@dataclass
class Result:
    """Outcome of checking one provider, including any issue that was filed."""

    provider: Provider
    current: Version | None = None
    latest: Latest | None = None
    locations: list[Location] = field(default_factory=list)
    error: str = ""
    issue_action: str = ""
    issue_url: str = ""

    @property
    def outdated(self) -> bool:
        return self.current is not None and self.latest is not None and self.latest.version.key > self.current.key


class GitHubApi:
    """Minimal stdlib GitHub REST client."""

    def __init__(self, token: str | None) -> None:
        self._token = token

    def request(
        self, method: str, path: str, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None
    ) -> Any:
        url = f"{API_ROOT}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ogx-check-provider-versions",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"Failed to {method} {path}: HTTP {e.code} {detail}") from e


def read_current_pin(repo_root: Path, provider: Provider) -> Version:
    for source in provider.pin_sources:
        path = repo_root / source.path
        if not path.exists():
            continue
        match = re.search(source.pattern, path.read_text(), re.MULTILINE)
        if match is not None:
            group = match.group(1)
            return parse_build(group) if provider.key == "llamacpp" else parse_version(group)
    searched = ", ".join(source.path for source in provider.pin_sources)
    raise RuntimeError(f"Failed to find the {provider.display} version pin in: {searched}")


def find_locations(repo_root: Path, provider: Provider, current: Version) -> list[Location]:
    needle = re.escape(current.label)
    pattern = rf"(?<![\w.]){needle}(?![\w.])"
    if provider.extra_location_pattern:
        pattern += f"|{provider.extra_location_pattern}"
    regex = re.compile(pattern)
    locations = []
    for rel_path in provider.location_files:
        path = repo_root / rel_path
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if regex.search(line):
                locations.append(Location(rel_path, line_number, line.strip()))
    return locations


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def select_blessed_build(releases: list[dict[str, Any]], semver_published_at: str) -> dict[str, Any] | None:
    """Latest bNNNN build published on or before the semver release."""
    cutoff = _parse_timestamp(semver_published_at)
    candidates = []
    for release in releases:
        match = BUILD_TAG_RE.match(release["tag_name"])
        if match is None or release.get("draft"):
            continue
        published = _parse_timestamp(release["published_at"])
        if published <= cutoff:
            candidates.append((published, int(match.group(1)), release))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def _digest_sha256(asset: dict[str, Any]) -> str:
    digest = asset.get("digest") or ""
    return digest.removeprefix("sha256:") if digest.startswith("sha256:") else ""


def resolve_latest_simple(api: GitHubApi, provider: Provider) -> Latest:
    release = api.request("GET", f"/repos/{provider.upstream}/releases/latest")
    return Latest(
        version=parse_version(release["tag_name"]),
        url=release["html_url"],
        published_at=release["published_at"],
    )


def resolve_latest_llamacpp(api: GitHubApi, provider: Provider) -> Latest:
    semver_release = api.request("GET", f"/repos/{provider.upstream}/releases/latest")
    semver_published_at = semver_release["published_at"]

    blessed = None
    for page in range(1, LLAMACPP_MAX_PAGES + 1):
        releases = api.request("GET", f"/repos/{provider.upstream}/releases", {"per_page": 100, "page": page})
        blessed = select_blessed_build(releases, semver_published_at)
        if blessed is not None or not releases:
            break
    if blessed is None:
        raise RuntimeError(
            f"Failed to find a llama.cpp build published on or before {semver_release['tag_name']} "
            f"({semver_published_at})"
        )

    tarball_name = LLAMACPP_ASSET_TEMPLATE.format(tag=blessed["tag_name"])
    asset = next((a for a in blessed.get("assets", []) if a["name"] == tarball_name), None)
    return Latest(
        version=parse_build(blessed["tag_name"]),
        url=blessed["html_url"],
        published_at=blessed["published_at"],
        release_tag=semver_release["tag_name"],
        tarball_name=tarball_name,
        tarball_sha256=_digest_sha256(asset) if asset else "",
    )


LATEST_RESOLVERS: dict[str, Callable[[GitHubApi, Provider], Latest]] = {
    "vllm": resolve_latest_simple,
    "ollama": resolve_latest_simple,
    "llamacpp": resolve_latest_llamacpp,
}


def latest_label(result: Result) -> str:
    latest = result.latest
    if latest is None:
        return "unknown"
    if latest.release_tag:
        return f"{latest.version.label} ({latest.release_tag})"
    return f"v{latest.version.label}"


def current_label(result: Result) -> str:
    if result.current is None:
        return "unknown"
    return result.current.label if result.provider.key == "llamacpp" else f"v{result.current.label}"


def issue_title(result: Result) -> str:
    return f"Update {result.provider.display} test server pin to {latest_label(result)}"


RE_RECORD_NOTE = (
    "Recordings are keyed by a hash of the request body, and the automatic record workflow only fills in "
    "*missing* recordings, so refreshing existing ones needs a forced re-record "
    '(`--inference-mode record`, see "Re-recording Tests" in `tests/integration/README.md`).'
)


def procedure(result: Result, repo_root: Path) -> str:
    key = result.provider.key
    latest = result.latest
    assert latest is not None
    if key == "vllm":
        if (repo_root / ".github/workflows/record-vllm-gpu-tests.yml").exists():
            record = (
                "2. Run the **vLLM GPU Recording** workflow (`record-vllm-gpu-tests.yml`) via workflow dispatch for "
                "the `base`, `responses` and `vllm-reasoning` suites, passing the PR number so recordings are "
                "committed back to the PR."
            )
        else:
            record = (
                "2. The **Integration Tests (Record)** workflow records the `vllm` setup automatically on the PR. "
                + RE_RECORD_NOTE
            )
        return (
            "1. Update the version everywhere listed above. If the setup pins a wheel checksum "
            "(`WHEEL_SHA`), recompute it for the new release asset.\n"
            f"{record}\n"
            "3. Confirm the recordings replay green in CI and note any provider-facing behavior changes."
        )
    if key == "ollama":
        return (
            "1. Bump the `ollama-version` default in `.github/actions/setup-ollama/action.yml` and the `FROM` line "
            "of both containerfiles under `tests/containers/`. The Docker Hub tag has no leading `v`.\n"
            "2. Re-record the Ollama suites (`base`, `vision`, `ollama-reasoning`, `messages`). "
            f"{RE_RECORD_NOTE}\n"
            "3. Rebuild and publish the all-in-one Ollama images from the containerfiles if they are consumed "
            "outside this repository."
        )
    sha = latest.tarball_sha256
    sha_line = (
        f"`{sha}`" if sha else "_not available from the release asset API; compute it with `sha256sum` after download_"
    )
    return (
        f"1. In `.github/actions/setup-llamacpp/action.yml`, replace every `{result.current.label if result.current else ''}` "
        f"with `{latest.version.label}` (download URL, extracted directory, server path) and set "
        f"`LLAMA_TARBALL_SHA` to the checksum below.\n"
        f"   - `{latest.tarball_name}` sha256: {sha_line}\n"
        "2. Re-record the `llama-cpp-server` suite. " + RE_RECORD_NOTE + "\n"
        "3. Check that the router preset options still match the new `llama-server` flags."
    )


def _code(text: str) -> str:
    """Inline code span that survives backticks in the text."""
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def render_issue_body(result: Result, repo_root: Path, run_url: str) -> str:
    latest = result.latest
    assert latest is not None
    rows = [
        "| | |",
        "| --- | --- |",
        f"| Provider | {result.provider.display} |",
        f"| Pinned | `{current_label(result)}` |",
        f"| Latest | [{latest_label(result)}]({latest.url}) (published {latest.published_at[:10]}) |",
    ]
    if latest.release_tag:
        rows.append(f"| Upstream release | `{latest.release_tag}` (blessed build: `{latest.version.label}`) |")
    locations = "\n".join(f"- `{loc.path}:{loc.line}`: {_code(loc.text)}" for loc in result.locations) or (
        "_No lines found automatically. Search the repository for the pinned version._"
    )
    footer = f"_Opened automatically by the [Check Provider Versions]({run_url}) workflow._" if run_url else ""
    return (
        f"Integration test recordings are produced against a live {result.provider.display} server, and a newer "
        "upstream release is available.\n\n"
        + "\n".join(rows)
        + f"\n\n## Where to update\n\n{locations}\n\n## Procedure\n\n{procedure(result, repo_root)}\n\n{footer}\n"
    )


def find_existing_issue(api: GitHubApi, target_repo: str, title: str) -> dict[str, Any] | None:
    query = f'repo:{target_repo} is:issue in:title "{title}"'
    found = api.request("GET", "/search/issues", {"q": query, "per_page": 30})
    return next((item for item in found.get("items", []) if item["title"] == title), None)


def file_issue(api: GitHubApi, target_repo: str, result: Result, repo_root: Path, run_url: str, dry_run: bool) -> None:
    title = issue_title(result)
    existing = find_existing_issue(api, target_repo, title)
    if existing is not None:
        result.issue_action = "exists"
        result.issue_url = existing["html_url"]
        return
    if dry_run:
        result.issue_action = "would create"
        return
    body = render_issue_body(result, repo_root, run_url)
    created = api.request("POST", f"/repos/{target_repo}/issues", body={"title": title, "body": body})
    result.issue_action = "created"
    result.issue_url = created["html_url"]


def check_provider(api: GitHubApi, provider: Provider, repo_root: Path) -> Result:
    result = Result(provider=provider)
    try:
        result.current = read_current_pin(repo_root, provider)
        result.latest = LATEST_RESOLVERS[provider.key](api, provider)
        result.locations = find_locations(repo_root, provider, result.current)
    except (RuntimeError, ValueError) as e:
        result.error = str(e)
    return result


def render_summary(results: list[Result]) -> str:
    lines = ["## Provider server versions", "", "| Provider | Pinned | Latest | Status |", "| --- | --- | --- | --- |"]
    for result in results:
        if result.error:
            status = f"error: {result.error}"
        elif not result.outdated:
            status = "up to date"
        elif result.issue_url:
            status = f"update available ([issue]({result.issue_url}), {result.issue_action})"
        else:
            status = f"update available ({result.issue_action or 'no issue filed'})"
        lines.append(f"| {result.provider.display} | `{current_label(result)}` | `{latest_label(result)}` | {status} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--target-repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--dry-run", action="store_true", help="Report what would be filed without creating issues")
    args = parser.parse_args(argv)

    if not args.target_repo:
        parser.error("--target-repo is required when GITHUB_REPOSITORY is not set")

    api = GitHubApi(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    server, run_id = os.environ.get("GITHUB_SERVER_URL", ""), os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"{server}/{args.target_repo}/actions/runs/{run_id}" if server and run_id else ""

    results = []
    for provider in PROVIDERS:
        result = check_provider(api, provider, args.repo_root)
        if result.outdated and not result.error:
            try:
                file_issue(api, args.target_repo, result, args.repo_root, run_url, args.dry_run)
            except RuntimeError as e:
                result.error = str(e)
        results.append(result)

    summary = render_summary(results)
    print(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary)

    failed = [result for result in results if result.error]
    for result in failed:
        print(f"::error::{result.provider.display}: {result.error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
