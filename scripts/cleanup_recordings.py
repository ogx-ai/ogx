#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Clean up unused test recordings based on CI test collection.

This script:
1. Reads CI matrix definitions from tests/integration/ci_matrix.json (default + scheduled overrides)
2. Uses pytest --collect-only with --json-report to gather all test IDs that run in CI,
   running the per-config collections concurrently
3. Compares against existing recordings to identify unused ones
4. Checks model-list recordings (models-*.json) for stale hash formats and
   redundant model-set digests
5. Optionally deletes unused recordings

Model-list recordings are special: replay unions every ``models-{hash}-{digest}.json``
file sharing a request hash, so a digest file is redundant when the kept files for
the same (directory, request hash) still cover every model ID the test suites
reference. A file whose filename hash no longer matches the current hash function
is dead: no replay lookup can ever find it.

Usage:
    # Dry run - see what would be deleted
    ./scripts/cleanup_recordings.py

    # Save manifest of CI test IDs for inspection
    ./scripts/cleanup_recordings.py --manifest ci_tests.txt

    # Actually delete unused recordings
    ./scripts/cleanup_recordings.py --delete

    # Control how many configs collect concurrently (default: min(cpu_count, 8))
    ./scripts/cleanup_recordings.py --jobs 4

    # Only check model-list recordings (skips CI test collection; seconds instead of minutes)
    ./scripts/cleanup_recordings.py --model-lists-only
"""

import argparse
import concurrent.futures
import json
import os
import re
import runpy
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Load CI matrix from JSON file
CI_MATRIX_FILE = REPO_ROOT / "tests/integration/ci_matrix.json"
with open(CI_MATRIX_FILE) as f:
    _matrix_config = json.load(f)

DEFAULT_CI_MATRIX: list[dict[str, str]] = _matrix_config["default"]
SCHEDULED_MATRICES: dict[str, list[dict[str, str]]] = _matrix_config.get("schedules", {})

# Top-level ``test_id`` is always written as the first key (see store_recording),
# so a short file prefix is enough to read it without parsing the response body.
_TEST_ID_RE = re.compile(rb'"test_id"\s*:\s*("(?:[^"\\]|\\.)*"|null)')

# Model-list recordings are named ``models-{request_hash}-{digest}.json`` (see
# store_recording). The digest covers the recorded model set; replay unions all
# digests of one request hash, so multiple digests per hash are expected.
MODEL_LIST_FILE_RE = re.compile(r"^models-([0-9a-f]{64})-([0-9a-f]{8})\.json$")

# Request-body keys that carry a model reference (covers OpenAI-compatible and
# genai-style provider payloads observed in the recordings).
_MODEL_REF_KEYS = ("model", "model_id")

try:
    import yaml

    from ogx.core.testing_context import set_test_context
    from ogx.testing.api_recorder import normalize_inference_request

    _RECORDER_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover - depends on project env
    yaml = None
    set_test_context = None
    normalize_inference_request = None
    _RECORDER_IMPORT_ERROR = _exc


def _unique_configs(entries):
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        suite = entry["suite"]
        setup = entry["setup"]
        key = (suite, setup)
        if key in seen:
            continue
        seen.add(key)
        yield {"suite": suite, "setup": setup}


def iter_all_ci_configs() -> list[dict[str, str]]:
    """Return unique CI configs across default and scheduled matrices.

    Also includes the suites that default to matrix-excluded setups (e.g. the
    k8s auth lane): those runs never appear in ci_matrix.json but their
    recordings must still count as referenced.
    """
    combined = list(DEFAULT_CI_MATRIX)
    for configs in SCHEDULED_MATRICES.values():
        combined.extend(configs)
    # runpy avoids depending on how this script is launched (sys.path may not
    # include the repo root); same pattern as generate_target_models_docs.py.
    suites_data = runpy.run_path(str(REPO_ROOT / "tests" / "integration" / "suites.py"))
    setups: dict[str, object] = suites_data["SETUP_DEFINITIONS"]
    for suite in suites_data["SUITE_DEFINITIONS"].values():
        default_setup = suite.default_setup
        if default_setup is None:
            continue
        setup = setups.get(default_setup)
        if setup is not None and getattr(setup, "matrix_excluded", False):
            combined.append({"suite": suite.name, "setup": default_setup})
    return list(_unique_configs(combined))


def _collect_config(config: dict[str, str]) -> tuple[set[str], int, str]:
    """Run one ``pytest --collect-only`` subprocess and parse its test node IDs.

    Returns ``(node_ids, returncode, message)``. ``message`` holds the failure
    detail when ``returncode != 0``, or a parse warning otherwise.
    """
    # Each config gets its own report file so concurrent runs never collide.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_report_file = f.name

    try:
        env = os.environ.copy()
        env["PYTEST_ADDOPTS"] = f"--json-report --json-report-file={json_report_file}"
        repo_path = str(REPO_ROOT)
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_path}{os.pathsep}{existing_path}" if existing_path else repo_path

        result = subprocess.run(
            [
                "./scripts/integration-tests.sh",
                "--collect-only",
                "--suite",
                config["suite"],
                "--setup",
                config["setup"],
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        if result.returncode != 0:
            return (
                set(),
                result.returncode,
                "Test collection failed.\n"
                f"Command: {' '.join(result.args)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}",
            )

        try:
            with open(json_report_file) as f:
                report = json.load(f)

            node_ids: set[str] = set()
            # The "collectors" field contains collected test items; each collector
            # has a "result" array with test node IDs.
            for collector in report.get("collectors", []):
                for item in collector.get("result", []):
                    if "nodeid" in item:
                        node_ids.add(item["nodeid"])
            return node_ids, 0, ""
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return set(), 0, f"Failed to parse JSON report: {e}"
    finally:
        if os.path.exists(json_report_file):
            os.unlink(json_report_file)


def collect_ci_tests(jobs: int) -> tuple[set[str], list[dict[str, str]]]:
    """Collect all test IDs that would run in CI using --collect-only with JSON output.

    Each config is a separate pytest subprocess; the collection work is CPU-bound,
    so running ``jobs`` of them concurrently overlaps the per-config cost.
    """
    all_test_ids: set[str] = set()
    configs = iter_all_ci_configs()
    print(f"Collecting tests for {len(configs)} CI configs with {jobs} concurrent job(s)...")

    hard_failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {executor.submit(_collect_config, config): config for config in configs}
        for future in concurrent.futures.as_completed(futures):
            config = futures[future]
            node_ids, returncode, message = future.result()
            all_test_ids |= node_ids
            if returncode != 0:
                hard_failures.append(f"suite={config['suite']}, setup={config['setup']}\n{message}")
            elif message:
                print(f"  Warning (suite={config['suite']}, setup={config['setup']}): {message}")
            print(
                f"  suite={config['suite']}, setup={config['setup']}: {len(node_ids)} IDs (total {len(all_test_ids)})"
            )

    if hard_failures:
        raise RuntimeError("Test collection failed for:\n\n" + "\n\n".join(hard_failures))

    print(f"\nTotal unique test IDs collected: {len(all_test_ids)}")
    return all_test_ids, configs


def get_base_test_id(test_id: str) -> str:
    """Extract base test ID without parameterization.

    Example:
      'tests/integration/inference/test_foo.py::test_bar[param1-param2]'
      -> 'tests/integration/inference/test_foo.py::test_bar'
    """
    return test_id.split("[")[0] if "[" in test_id else test_id


def find_all_recordings():
    """Find all recording JSON files."""
    return list((REPO_ROOT / "tests/integration").rglob("recordings/*.json"))


def is_model_list_file(json_file: Path) -> bool:
    """Return whether a recording file is a model-list recording."""
    return MODEL_LIST_FILE_RE.match(json_file.name) is not None


def extract_model_ids_from_list(body) -> frozenset[str]:
    """Extract the model identifiers from a recorded model-list response body.

    Handles the two recorded shapes: OpenAI-style lists of model objects and the
    TEI ``/info`` single-model response.
    """
    if isinstance(body, dict):
        data = body.get("__data__", body)
        if isinstance(data, dict) and (data.get("model_id") or data.get("id")):
            return frozenset({data.get("model_id") or data.get("id")})
        return frozenset()
    if not isinstance(body, list):
        return frozenset()
    ids: set[str] = set()
    for item in body:
        if not isinstance(item, dict):
            continue
        data = item.get("__data__", item)
        if isinstance(data, dict) and (data.get("id") or data.get("model_id")):
            ids.add(data.get("id") or data.get("model_id"))
    return frozenset(ids)


def _walk_request_model_ids(node, found: set[str]) -> None:
    """Collect string values of model-reference keys anywhere in a recording request."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _MODEL_REF_KEYS and isinstance(value, str) and value:
                found.add(value)
            else:
                _walk_request_model_ids(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_request_model_ids(item, found)


def _walk_config_model_ids(node, found: set[str]) -> None:
    """Collect registered model identifiers from a stack config tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("model_id", "provider_model_id") and isinstance(value, str) and value != "auto":
                found.add(value)
            else:
                _walk_config_model_ids(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_config_model_ids(item, found)


def collect_referenced_model_ids() -> set[str]:
    """Collect the model IDs the test suites actually reference.

    Sources:
    - ``model``/``model_id`` fields in the requests of every non-model-list
      recording (covers both client-level and server-side provider-level calls
      made by any recorded run, in any setup).
    - ``model_id``/``provider_model_id`` values registered in the CI stack configs.
    """
    referenced: set[str] = set()

    recording_files = [p for p in find_all_recordings() if not is_model_list_file(p)]
    print(f"Scanning {len(recording_files)} recordings for referenced model IDs...")
    for json_file in recording_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        _walk_request_model_ids(data.get("request", {}), referenced)

    config_dir = REPO_ROOT / "src/ogx/distributions/ci-tests"
    for config_file in sorted(config_dir.glob("*.yaml")):
        try:
            config = yaml.safe_load(config_file.read_text())
        except (OSError, yaml.YAMLError):
            continue
        _walk_config_model_ids(config, referenced)

    print(f"Referenced model IDs: {len(referenced)}")
    return referenced


def expected_model_list_hash(json_file: Path) -> str | None:
    """Recompute the request hash the current recorder computes for a stored model-list request.

    Returns None if the file cannot be parsed or the recorder imports are missing.
    """
    if normalize_inference_request is None:
        return None
    try:
        with open(json_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    request = data.get("request", {})
    set_test_context(data.get("test_id"))
    try:
        return normalize_inference_request(
            request.get("method", "POST"),
            request.get("url", ""),
            request.get("headers", {}) or {},
            request.get("body", {}) or {},
        )
    finally:
        set_test_context(None)


def analyze_model_list_recordings(
    model_files: list[Path], referenced_model_ids: set[str]
) -> tuple[list[Path], list[Path]]:
    """Classify model-list recordings as legacy or redundant.

    - Legacy: the filename hash does not match the hash the current recorder
      computes for the stored request. The hash function has changed since the
      file was written, so no replay lookup can ever find the file.
    - Redundant: the file's model set adds nothing the kept files of the same
      (directory, request hash) group do not already cover, given that replay
      unions all files of a group and only referenced model IDs matter.

    Every group keeps at least one file so the model-list request itself stays
    replayable. Returns ``(legacy_files, redundant_files)``.
    """
    legacy: list[Path] = []
    groups: dict[tuple[Path, str], list[tuple[Path, frozenset[str]]]] = defaultdict(list)

    for json_file in model_files:
        filename_hash = MODEL_LIST_FILE_RE.match(json_file.name).group(1)
        current_hash = expected_model_list_hash(json_file)
        if current_hash is None:
            raise RuntimeError(f"Failed to recompute hash for model-list recording {json_file.name}")
        if current_hash != filename_hash:
            legacy.append(json_file)
            continue
        try:
            with open(json_file) as f:
                body = json.load(f)["response"]["body"]
        except (OSError, json.JSONDecodeError, KeyError):
            # Unparseable files are left in place; the parse-error pass reports them.
            groups[(json_file.parent, filename_hash)].append((json_file, frozenset()))
            continue
        groups[(json_file.parent, filename_hash)].append((json_file, extract_model_ids_from_list(body)))

    redundant: list[Path] = []
    for (directory, _request_hash), entries in sorted(groups.items()):
        union = set().union(*(ids for _, ids in entries))
        needed = union & referenced_model_ids

        # Greedy set cover over the referenced models, preferring larger sets so
        # the most recent (superset) model sets are the ones kept. Ties break on
        # path so the result is stable across machines and filesystems.
        remaining = set(needed)
        pool = {path: set(ids) for path, ids in entries}
        kept: set[Path] = set()
        while remaining:
            path = max(sorted(pool), key=lambda p: (len(pool[p] & remaining), len(pool[p])))
            kept.add(path)
            remaining -= pool[path]
            del pool[path]

        # A model-list request for this hash must still be replayable even when
        # none of its models are referenced.
        if not kept:
            sizes = {path: len(ids) for path, ids in entries}
            kept.add(min((path for path, _ in entries), key=lambda p: (-sizes[p], str(p))))

        kept_union = set().union(*(ids for path, ids in entries if path in kept))
        if not needed <= kept_union:
            missing = sorted(needed - kept_union)
            raise RuntimeError(f"Failed to preserve model coverage for {directory / _request_hash[:12]}: {missing[:5]}")
        redundant.extend(path for path, _ in entries if path not in kept)

    return legacy, redundant


# Tool-call recordings store a request of ``{test_id, provider, tool_name, kwargs}``
# (no ``method``) and are replayed by a hash over (provider, tool_name, kwargs)
# alone, with no test_id. Any test issuing the same call can replay them, and the
# call kwargs are model-generated and unpredictable, so a stale test_id does not
# make a tool-call recording unused.
_TOOL_PROVIDER_RE = re.compile(rb'"provider"\s*:\s*"')
_TOOL_METHOD_RE = re.compile(rb'"method"\s*:')


def _is_tool_call_recording(json_file: Path) -> bool:
    """Return whether a recording is a tool-call recording, shared across tests."""
    with open(json_file, "rb") as f:
        prefix = f.read(65536)
    return bool(_TOOL_PROVIDER_RE.search(prefix)) and _TOOL_METHOD_RE.search(prefix) is None


def _read_test_id(json_file: Path) -> str | None:
    """Return a recording's top-level ``test_id`` without parsing its body.

    Only a short prefix is read, which matters for large streaming recordings.
    Falls back to a full parse if the key is not found in the prefix.
    """
    with open(json_file, "rb") as f:
        match = _TEST_ID_RE.search(f.read(65536))
    if match is not None:
        try:
            value: object = json.loads(match.group(1))
        except ValueError:
            value = None
        return value if isinstance(value, str) else None
    with open(json_file) as f:
        value = json.load(f).get("test_id", "")
    return value if isinstance(value, str) else None


def analyze_recordings(ci_test_ids, dry_run=True, model_only=False):
    """Analyze recordings and identify unused ones."""

    # Use full test IDs with parameterization for exact matching
    all_recordings = find_all_recordings()
    print(f"\nTotal recording files: {len(all_recordings)}")

    model_files = [p for p in all_recordings if is_model_list_file(p)]
    regular_files = [] if model_only else [p for p in all_recordings if not is_model_list_file(p)]

    # Categorize recordings
    used_recordings = []
    unused_recordings = []
    shared_recordings = []
    parse_errors = []

    for json_file in regular_files:
        try:
            test_id = _read_test_id(json_file)

            if not test_id:
                # Shared/infrastructure recordings
                shared_recordings.append(json_file)
                continue

            if _is_tool_call_recording(json_file):
                shared_recordings.append(json_file)
                continue

            # Match exact test_id (with full parameterization)
            if test_id in ci_test_ids:
                used_recordings.append(json_file)
            else:
                unused_recordings.append((json_file, test_id))

        except Exception as e:
            parse_errors.append((json_file, str(e)))

    # Model-list recordings follow their own rules: replay unions all digests of
    # one request hash, so they are checked for stale hash formats and redundant
    # model sets instead of being matched by test ID.
    legacy_model_files: list[Path] = []
    redundant_model_files: list[Path] = []
    if model_files:
        if normalize_inference_request is None:
            raise RuntimeError(
                f"Failed to import ogx.testing.api_recorder ({_RECORDER_IMPORT_ERROR}). "
                "Run via 'uv run --group dev' from the repo root so the project environment is used."
            )
        referenced_model_ids = collect_referenced_model_ids()
        legacy_model_files, redundant_model_files = analyze_model_list_recordings(model_files, referenced_model_ids)

    # Print summary
    print("\nRecording Analysis:")
    if not model_only:
        print(f"  Used in CI:        {len(used_recordings)}")
    print(f"  Model-list:        {len(model_files)}")
    if not model_only:
        print(f"  Shared (no ID or tool-call): {len(shared_recordings)}")
    print(f"  UNUSED:            {len(unused_recordings) + len(legacy_model_files) + len(redundant_model_files)}")
    if not model_only:
        print(f"    stale test ID:           {len(unused_recordings)}")
    print(f"    legacy model-list hash:  {len(legacy_model_files)}")
    print(f"    redundant model digest:  {len(redundant_model_files)}")
    if not model_only:
        print(f"  Parse errors:      {len(parse_errors)}")

    if unused_recordings:
        print("\nUnused recordings by test:")

        # Group by base test ID
        by_test = defaultdict(list)
        for file, test_id in unused_recordings:
            base = get_base_test_id(test_id)
            by_test[base].append(file)

        for base_test, files in sorted(by_test.items()):
            print(f"\n  {base_test}")
            print(f"    ({len(files)} recording(s))")
            for f in files[:3]:
                print(f"      - {f.relative_to(REPO_ROOT / 'tests/integration')}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more")

    if legacy_model_files or redundant_model_files:
        print("\nUnused model-list recordings:")
        for reason, files in (
            ("legacy hash (unreachable by replay)", legacy_model_files),
            ("redundant model-set digest", redundant_model_files),
        ):
            if not files:
                continue
            by_dir = defaultdict(list)
            for f in files:
                by_dir[f.parent].append(f)
            print(f"\n  {reason}: {len(files)}")
            for directory, dir_files in sorted(by_dir.items()):
                print(f"    {directory.relative_to(REPO_ROOT / 'tests/integration')}: {len(dir_files)}")
                for f in dir_files[:3]:
                    print(f"      - {f.relative_to(REPO_ROOT / 'tests/integration')}")
                if len(dir_files) > 3:
                    print(f"      ... and {len(dir_files) - 3} more")

    if parse_errors:
        print("\nParse errors:")
        for file, error in parse_errors[:5]:
            print(f"  {file.relative_to(REPO_ROOT)}: {error}")
        if len(parse_errors) > 5:
            print(f"  ... and {len(parse_errors) - 5} more")

    # Perform cleanup
    unused_all = [(f, "stale test ID") for f, _ in unused_recordings]
    unused_all += [(f, "legacy model-list hash") for f in legacy_model_files]
    unused_all += [(f, "redundant model-list digest") for f in redundant_model_files]

    if not dry_run:
        print(f"\nDeleting {len(unused_all)} unused recordings...")
        for file, _ in unused_all:
            file.unlink()
            print(f"  Deleted: {file.relative_to(REPO_ROOT / 'tests/integration')}")
        print("✅ Cleanup complete")
    else:
        print("\n(Dry run - no files deleted)")
        print("\nTo delete these files, run with --delete")

    return len(unused_all)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up unused test recordings based on CI test collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete unused recordings (default is dry-run)")
    parser.add_argument("--manifest", help="Save collected test IDs to file (optional)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of configs to collect concurrently (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--model-lists-only",
        action="store_true",
        help="Only check model-list recordings (skips CI test collection; seconds instead of minutes)",
    )

    args = parser.parse_args()

    if args.model_lists_only and args.manifest:
        parser.error("--manifest is not supported with --model-lists-only")

    print("=" * 60)
    print("Recording Cleanup Utility")
    print("=" * 60)

    if args.model_lists_only:
        unused_count = analyze_recordings(None, dry_run=not args.delete, model_only=True)
    else:
        ci_configs = iter_all_ci_configs()

        print(f"\nDetected CI configurations: {len(ci_configs)}")
        for config in ci_configs:
            print(f"  - suite={config['suite']}, setup={config['setup']}")

        # Collect test IDs from CI configurations
        ci_test_ids, _ = collect_ci_tests(args.jobs)

        if args.manifest:
            with open(args.manifest, "w") as f:
                for test_id in sorted(ci_test_ids):
                    f.write(f"{test_id}\n")
            print(f"\nSaved test IDs to: {args.manifest}")

        # Analyze and cleanup
        unused_count = analyze_recordings(ci_test_ids, dry_run=not args.delete)

    print("\n" + "=" * 60)
    if unused_count > 0 and not args.delete:
        print("Run with --delete to remove unused recordings")


if __name__ == "__main__":
    main()
