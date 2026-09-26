#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Resolve which Claude Code CLI and Agent SDK versions the messages client smoke tests install.

Merge CI tests the `stable` channel and the nightly run tests `latest`, so a client release
that breaks /v1/messages is caught nightly without blocking unrelated PRs on release day.

The CLI has real channels: https://downloads.claude.ai/claude-code-releases/{stable,latest}
each return a bare version number.

PyPI has no channels for the Agent SDK, so the equivalent is derived from PyPI's own data:
`latest` is the newest release and `stable` is the newest release that is at least
``--sdk-stable-min-age-days`` old, which mirrors the soak time of the CLI's stable channel.

Prints the resolved versions and, when GITHUB_OUTPUT is set, appends them there as
``cli_version`` and ``sdk_version``.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

CLI_CHANNEL_URL = "https://downloads.claude.ai/claude-code-releases/{channel}"
SDK_PYPI_URL = "https://pypi.org/pypi/claude-agent-sdk/json"
CHANNELS = ("stable", "latest")
DEFAULT_SDK_STABLE_MIN_AGE_DAYS = 7

# Plain releases only: no rc, dev or post suffixes.
RELEASE_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ogx-resolve-claude-client-versions"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            body: str = response.read().decode()
            return body
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}") from e


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _require_release(version: str, what: str) -> str:
    if not RELEASE_RE.match(version):
        raise ValueError(f"Failed to parse {what}: expected a version like 1.2.3, got {version[:80]!r}")
    return version


def resolve_cli_version(channel: str, fetch: Callable[[str], str] = _fetch) -> str:
    """The version the Claude Code CLI's ``channel`` currently points at."""
    if channel not in CHANNELS:
        raise ValueError(f"Failed to resolve CLI version: unknown channel {channel!r}, expected one of {CHANNELS}")
    return _require_release(fetch(CLI_CHANNEL_URL.format(channel=channel)).strip(), f"the {channel} CLI version")


def _usable_releases(pypi: dict[str, Any]) -> list[tuple[str, datetime]]:
    """(version, upload time) of every plain, non-yanked release that has files."""
    releases = []
    for version, files in pypi.get("releases", {}).items():
        if not RELEASE_RE.match(version) or not files or all(f.get("yanked") for f in files):
            continue
        uploaded = datetime.fromisoformat(min(f["upload_time_iso_8601"] for f in files).replace("Z", "+00:00"))
        releases.append((version, uploaded))
    return releases


def resolve_sdk_version(
    channel: str,
    pypi: dict[str, Any],
    now: datetime | None = None,
    stable_min_age_days: int = DEFAULT_SDK_STABLE_MIN_AGE_DAYS,
) -> str:
    """The Agent SDK release for ``channel``: the newest, or for stable the newest old enough."""
    if channel not in CHANNELS:
        raise ValueError(f"Failed to resolve SDK version: unknown channel {channel!r}, expected one of {CHANNELS}")
    releases = _usable_releases(pypi)
    if channel == "stable":
        cutoff = (now or datetime.now(UTC)) - timedelta(days=stable_min_age_days)
        releases = [(version, uploaded) for version, uploaded in releases if uploaded <= cutoff]
    if not releases:
        raise RuntimeError(f"Failed to find a {channel} claude-agent-sdk release on PyPI")
    return max((version for version, _ in releases), key=_version_key)


def resolve(
    channel: str,
    cli_override: str = "",
    sdk_override: str = "",
    fetch: Callable[[str], str] = _fetch,
    now: datetime | None = None,
    stable_min_age_days: int = DEFAULT_SDK_STABLE_MIN_AGE_DAYS,
) -> dict[str, str]:
    """Resolve both versions. An exact override skips the lookup, which allows bisecting a failure."""
    cli_version = (
        _require_release(cli_override, "the CLI version override")
        if cli_override
        else resolve_cli_version(channel, fetch)
    )
    if sdk_override:
        sdk_version = _require_release(sdk_override, "the SDK version override")
    else:
        sdk_version = resolve_sdk_version(channel, json.loads(fetch(SDK_PYPI_URL)), now, stable_min_age_days)
    return {"cli_version": cli_version, "sdk_version": sdk_version}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", choices=CHANNELS, required=True)
    parser.add_argument("--cli-version", default="", help="Exact CLI version, overriding the channel")
    parser.add_argument("--sdk-version", default="", help="Exact SDK version, overriding the channel")
    parser.add_argument("--sdk-stable-min-age-days", type=int, default=DEFAULT_SDK_STABLE_MIN_AGE_DAYS)
    args = parser.parse_args(argv)

    try:
        versions = resolve(
            args.channel,
            args.cli_version,
            args.sdk_version,
            stable_min_age_days=args.sdk_stable_min_age_days,
        )
    except (RuntimeError, ValueError) as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1

    lines = [f"{key}={value}" for key, value in versions.items()]
    print(f"channel={args.channel}")
    print("\n".join(lines))
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
