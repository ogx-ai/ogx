#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Skills API Demo — exercises all 11 Skills API endpoints.

Prerequisites:
    Start an OGX server with skills enabled:
        uv run ogx letsgo

Usage:
    uv run python demo/skills_api_demo.py [--base-url http://localhost:8321]
"""

import argparse
import io
import json
import sys
import zipfile

import httpx

DEFAULT_BASE_URL = "http://localhost:8321"


def make_skill_zip(name: str, description: str, instructions: str, files: dict[str, str] | None = None) -> bytes:
    """Create a skill zip bundle with a SKILL.md manifest and optional extra files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = f"""---
name: {name}
description: {description}
---
{instructions}"""
        zf.writestr("SKILL.md", manifest)
        if files:
            for path, content in files.items():
                zf.writestr(path, content)
    return buf.getvalue()


def print_response(label: str, resp: httpx.Response) -> dict | None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Status: {resp.status_code}")
    if resp.headers.get("content-type", "").startswith("application/json"):
        data = resp.json()
        print(f"  Response: {json.dumps(data, indent=2)}")
        return data
    elif resp.headers.get("content-type", "").startswith("application/zip"):
        print(f"  Response: <zip archive, {len(resp.content)} bytes>")
        return None
    else:
        print(f"  Response: {resp.text[:200]}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Skills API Demo")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OGX server URL")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=30.0)

    print("\n" + "=" * 60)
    print("  OGX Skills API Demo")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 1. Create a skill
    # ──────────────────────────────────────────────────────────
    zip_content = make_skill_zip(
        name="data-analyzer",
        description="Analyzes CSV data and generates summary statistics",
        instructions="Use this skill to analyze CSV files. Run `python analyze.py <file>` to get statistics.",
        files={
            "analyze.py": 'import sys\nprint(f"Analyzing {sys.argv[1]}...")\nprint("Mean: 42.0, Median: 41.5")\n',
            "requirements.txt": "pandas>=2.0\n",
        },
    )

    resp = client.post(
        "/v1alpha/skills",
        files={"file": ("data-analyzer.zip", zip_content, "application/zip")},
    )
    skill = print_response("1. POST /v1alpha/skills — Create skill", resp)
    if resp.status_code != 200:
        print("\n  ERROR: Failed to create skill. Is the server running with skills enabled?")
        sys.exit(1)

    skill_id = skill["id"]

    # ──────────────────────────────────────────────────────────
    # 2. List skills
    # ──────────────────────────────────────────────────────────
    resp = client.get("/v1alpha/skills")
    print_response("2. GET /v1alpha/skills — List skills", resp)

    # ──────────────────────────────────────────────────────────
    # 3. Get skill metadata
    # ──────────────────────────────────────────────────────────
    resp = client.get(f"/v1alpha/skills/{skill_id}")
    print_response(f"3. GET /v1alpha/skills/{skill_id} — Get skill", resp)

    # ──────────────────────────────────────────────────────────
    # 4. Download skill content
    # ──────────────────────────────────────────────────────────
    resp = client.get(f"/v1alpha/skills/{skill_id}/content")
    print_response(f"4. GET /v1alpha/skills/{skill_id}/content — Download zip", resp)

    # ──────────────────────────────────────────────────────────
    # 5. Create a new version
    # ──────────────────────────────────────────────────────────
    zip_v2 = make_skill_zip(
        name="data-analyzer",
        description="Analyzes CSV data with visualization support",
        instructions="Run `python analyze.py <file>` for stats, or `python plot.py <file>` for charts.",
        files={
            "analyze.py": 'import sys\nprint(f"Analyzing {sys.argv[1]}...")\nprint("Mean: 42.0, Std: 5.2")\n',
            "plot.py": 'import sys\nprint(f"Plotting {sys.argv[1]}...")\n',
            "requirements.txt": "pandas>=2.0\nmatplotlib>=3.8\n",
        },
    )

    resp = client.post(
        f"/v1alpha/skills/{skill_id}/versions",
        files={"file": ("data-analyzer-v2.zip", zip_v2, "application/zip")},
        data={"default": "true"},
    )
    print_response(f"5. POST /v1alpha/skills/{skill_id}/versions — Create version 2", resp)

    # ──────────────────────────────────────────────────────────
    # 6. List versions
    # ──────────────────────────────────────────────────────────
    resp = client.get(f"/v1alpha/skills/{skill_id}/versions")
    print_response(f"6. GET /v1alpha/skills/{skill_id}/versions — List versions", resp)

    # ──────────────────────────────────────────────────────────
    # 7. Get version metadata
    # ──────────────────────────────────────────────────────────
    resp = client.get(f"/v1alpha/skills/{skill_id}/versions/2")
    print_response(f"7. GET /v1alpha/skills/{skill_id}/versions/2 — Get version 2", resp)

    # ──────────────────────────────────────────────────────────
    # 8. Download version content
    # ──────────────────────────────────────────────────────────
    resp = client.get(f"/v1alpha/skills/{skill_id}/versions/1/content")
    print_response(f"8. GET /v1alpha/skills/{skill_id}/versions/1/content — Download v1 zip", resp)

    # ──────────────────────────────────────────────────────────
    # 9. Update default version
    # ──────────────────────────────────────────────────────────
    resp = client.post(
        f"/v1alpha/skills/{skill_id}",
        json={"default_version": "1"},
    )
    print_response(f"9. POST /v1alpha/skills/{skill_id} — Set default to v1", resp)

    # ──────────────────────────────────────────────────────────
    # 10. Delete version 1
    # ──────────────────────────────────────────────────────────
    resp = client.delete(f"/v1alpha/skills/{skill_id}/versions/1")
    print_response(f"10. DELETE /v1alpha/skills/{skill_id}/versions/1 — Delete version 1", resp)

    # Verify default_version was updated
    resp = client.get(f"/v1alpha/skills/{skill_id}")
    updated = print_response(f"    GET /v1alpha/skills/{skill_id} — Verify default updated", resp)
    if updated:
        print(f"\n    default_version is now: {updated['default_version']}")

    # ──────────────────────────────────────────────────────────
    # 11. Delete skill
    # ──────────────────────────────────────────────────────────
    resp = client.delete(f"/v1alpha/skills/{skill_id}")
    print_response(f"11. DELETE /v1alpha/skills/{skill_id} — Delete skill", resp)

    # Verify deletion
    resp = client.get(f"/v1alpha/skills/{skill_id}")
    print_response(f"    GET /v1alpha/skills/{skill_id} — Verify deleted (expect error)", resp)

    print("\n" + "=" * 60)
    print("  Demo complete — all 11 Skills API endpoints exercised")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
