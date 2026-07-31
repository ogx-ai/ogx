#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Diagnostic tool for debugging test recording issues.

Usage:
    # Find where a hash would be looked up
    ./scripts/diagnose_recordings.py find-hash 7526c930eab04ce337496a26cd15f2591d7943035f2527182861643da9b837a7

    # Show what's in a recording file
    ./scripts/diagnose_recordings.py show tests/integration/agents/recordings/7526c930....json

    # List all recordings for a test
    ./scripts/diagnose_recordings.py list-test "tests/integration/agents/test_agents.py::test_custom_tool"

    # Explain lookup paths for a test
    ./scripts/diagnose_recordings.py explain-paths --test-id "tests/integration/agents/test_agents.py::test_foo"

    # Compare request hash computation
    ./scripts/diagnose_recordings.py compute-hash --endpoint /v1/chat/completions --method POST --body '{"model":"llama3.2:3b"}' --test-id "..."
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import from ogx
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from ogx.testing.api_recorder import normalize_inference_request
except ImportError:
    normalize_inference_request = None


def find_hash(hash_value: str, base_dir: Path | None = None, test_id: str | None = None):
    """Find where a hash would be looked up and what exists"""
    if base_dir is None:
        base_dir = REPO_ROOT / "tests/integration/common"

    print(f"Searching for hash: {hash_value}\n")
    print(f"Base dir: {base_dir} (absolute={base_dir.is_absolute()})")

    # Compute test directory
    if test_id:
        test_file = test_id.split("::")[0]
        test_dir = Path(test_file).parent

        if base_dir.is_absolute():
            repo_root = base_dir.parent.parent.parent
            test_recordings_dir = repo_root / test_dir / "recordings"
        else:
            test_recordings_dir = test_dir / "recordings"
        print(f"Test ID: {test_id}")
        print(f"Test dir: {test_recordings_dir}\n")
    else:
        test_recordings_dir = base_dir / "recordings"
        print("No test ID provided, using base dir\n")

    # Check primary location
    response_file = f"{hash_value}.json"
    response_path = test_recordings_dir / response_file

    print("Checking primary location:")
    print(f"  {response_path}")
    if response_path.exists():
        print("  EXISTS")
        print("\nFound! Contents:")
        show_recording(response_path)
        return True
    else:
        print("  Does not exist")

    # Check fallback location
    fallback_dir = base_dir / "recordings"
    fallback_path = fallback_dir / response_file

    print("\nChecking fallback location:")
    print(f"  {fallback_path}")
    if fallback_path.exists():
        print("  EXISTS")
        print("\nFound in fallback! Contents:")
        show_recording(fallback_path)
        return True
    else:
        print("  Does not exist")

    # Show what files DO exist
    print(f"\nFiles in test directory ({test_recordings_dir}):")
    if test_recordings_dir.exists():
        json_files = list(test_recordings_dir.glob("*.json"))
        if json_files:
            for f in json_files[:20]:
                print(f"  - {f.name}")
            if len(json_files) > 20:
                print(f"  ... and {len(json_files) - 20} more")
        else:
            print("  (empty)")
    else:
        print("  Directory does not exist")

    print(f"\nFiles in fallback directory ({fallback_dir}):")
    if fallback_dir.exists():
        json_files = list(fallback_dir.glob("*.json"))
        if json_files:
            for f in json_files[:20]:
                print(f"  - {f.name}")
            if len(json_files) > 20:
                print(f"  ... and {len(json_files) - 20} more")
        else:
            print("  (empty)")
    else:
        print("  Directory does not exist")

    # Try partial hash match
    print("\nLooking for partial matches (first 16 chars)...")
    partial = hash_value[:16]
    matches = []

    for dir_to_search in [test_recordings_dir, fallback_dir]:
        if dir_to_search.exists():
            for f in dir_to_search.glob("*.json"):
                if f.stem.startswith(partial):
                    matches.append(f)

    if matches:
        print(f"Found {len(matches)} partial match(es):")
        for m in matches:
            print(f"  {m}")
    else:
        print("No partial matches found")

    return False


def show_recording(file_path: Path):
    """Show contents of a recording file"""
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return

    with open(file_path) as f:
        data = json.load(f)

    print(f"\nRecording: {file_path.name}\n")
    print(f"Test ID: {data.get('test_id', 'N/A')}")
    print("\nRequest:")
    req = data.get("request", {})
    print(f"  Method: {req.get('method', 'N/A')}")
    print(f"  URL: {req.get('url', 'N/A')}")
    print(f"  Endpoint: {req.get('endpoint', 'N/A')}")
    print(f"  Model: {req.get('model', 'N/A')}")

    body = req.get("body", {})
    if body:
        print("\nRequest Body:")
        print(f"  Model: {body.get('model', 'N/A')}")
        print(f"  Stream: {body.get('stream', 'N/A')}")
        if "messages" in body:
            print(f"  Messages: {len(body['messages'])} message(s)")
            for i, msg in enumerate(body["messages"][:3]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    preview = content[:80] + "..." if len(content) > 80 else content
                else:
                    preview = f"[{type(content).__name__}]"
                print(f"    [{i}] {role}: {preview}")
        if "tools" in body:
            print(f"  Tools: {len(body['tools'])} tool(s)")

    response = data.get("response", {})
    if response:
        print("\nResponse:")
        print(f"  Is streaming: {response.get('is_streaming', False)}")
        response_body = response.get("body", {})
        if isinstance(response_body, dict):
            if "__type__" in response_body:
                print(f"  Type: {response_body['__type__']}")
            if "__data__" in response_body:
                response_data = response_body["__data__"]
                if "choices" in response_data:
                    print(f"  Choices: {len(response_data['choices'])}")
                if "usage" in response_data:
                    usage = response_data["usage"]
                    print(f"  Usage: in={usage.get('input_tokens')}, out={usage.get('output_tokens')}")


def list_test_recordings(test_id: str, base_dir: Path | None = None):
    """List all recordings for a specific test"""
    if base_dir is None:
        base_dir = REPO_ROOT / "tests/integration/common"

    test_file = test_id.split("::")[0]
    test_dir = Path(test_file).parent

    if base_dir.is_absolute():
        repo_root = base_dir.parent.parent.parent
        test_recordings_dir = repo_root / test_dir / "recordings"
    else:
        test_recordings_dir = test_dir / "recordings"

    print(f"Recordings for test: {test_id}\n")
    print(f"Directory: {test_recordings_dir}\n")

    if not test_recordings_dir.exists():
        print("Directory does not exist")
        return

    # Find all recordings for this specific test
    recordings = []
    for f in test_recordings_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                if data.get("test_id") == test_id:
                    recordings.append((f, data))
        except Exception as e:
            print(f"Could not read {f.name}: {e}")

    if not recordings:
        print("No recordings found for this exact test ID")
        print("\nAll files in directory:")
        for f in test_recordings_dir.glob("*.json"):
            print(f"  - {f.name}")
        return

    print(f"Found {len(recordings)} recording(s):\n")
    for f, data in recordings:
        req = data.get("request", {})
        print(f"  {f.name}")
        print(f"     Endpoint: {req.get('endpoint', 'N/A')}")
        print(f"     Model: {req.get('model', 'N/A')}")
        print("")


def explain_paths(test_id: str | None = None, base_dir: Path | None = None):
    """Explain where recordings would be searched"""
    if base_dir is None:
        base_dir = REPO_ROOT / "tests/integration/common"

    print("Recording Lookup Path Explanation\n")
    print(f"Base directory: {base_dir}")
    print(f"  Absolute: {base_dir.is_absolute()}")
    print("")

    if test_id:
        print(f"Test ID: {test_id}")
        test_file = test_id.split("::")[0]
        print(f"  Test file: {test_file}")

        test_dir = Path(test_file).parent
        print(f"  Test dir (relative): {test_dir}")

        if base_dir.is_absolute():
            repo_root = base_dir.parent.parent.parent
            print(f"  Repo root: {repo_root}")
            test_recordings_dir = repo_root / test_dir / "recordings"
            print(f"  Test recordings dir (absolute): {test_recordings_dir}")
        else:
            test_recordings_dir = test_dir / "recordings"
            print(f"  Test recordings dir (relative): {test_recordings_dir}")

        print("\nLookup order for recordings:")
        print(f"  1. Test-specific: {test_recordings_dir}/<hash>.json")
        print(f"  2. Fallback: {base_dir}/recordings/<hash>.json")

    else:
        print("No test ID provided")
        print("\nLookup location:")
        print(f"  {base_dir}/recordings/<hash>.json")


def compute_hash(endpoint: str, method: str, body_json: str, test_id: str | None = None):
    """Compute hash for a request"""
    if normalize_inference_request is None:
        print("Could not import normalize_inference_request from ogx.testing.api_recorder")
        print("Make sure you're running from the repo root with proper PYTHONPATH")
        return

    try:
        body = json.loads(body_json)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in body: {e}")
        return

    # Create a fake URL with the endpoint
    url = f"http://example.com{endpoint}"

    # Set test context if provided
    if test_id:
        from ogx.core.testing_context import set_test_context

        set_test_context(test_id)

    hash_result = normalize_inference_request(method, url, {}, body)

    print("Hash Computation\n")
    print(f"Method: {method}")
    print(f"Endpoint: {endpoint}")
    print(f"Test ID: {test_id or 'None (excluded from hash for model-list endpoints)'}")
    print("\nBody:")
    print(json.dumps(body, indent=2))
    print(f"\nComputed Hash: {hash_result}")
    print(f"\nLooking for file: {hash_result}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for test recording issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # find-hash command
    find_parser = subparsers.add_parser("find-hash", help="Find where a hash would be looked up")
    find_parser.add_argument("hash", help="Hash value to search for (full or partial)")
    find_parser.add_argument("--test-id", help="Test ID to determine search paths")
    find_parser.add_argument("--base-dir", type=Path, help="Base directory (default: tests/integration/common)")

    # show command
    show_parser = subparsers.add_parser("show", help="Show contents of a recording file")
    show_parser.add_argument("file", type=Path, help="Path to recording JSON file")

    # list-test command
    list_parser = subparsers.add_parser("list-test", help="List all recordings for a test")
    list_parser.add_argument("test_id", help="Full test ID (e.g., tests/integration/agents/test_agents.py::test_foo)")
    list_parser.add_argument("--base-dir", type=Path, help="Base directory (default: tests/integration/common)")

    # explain-paths command
    explain_parser = subparsers.add_parser("explain-paths", help="Explain where recordings are searched")
    explain_parser.add_argument("--test-id", help="Test ID to show paths for")
    explain_parser.add_argument("--base-dir", type=Path, help="Base directory (default: tests/integration/common)")

    # compute-hash command
    hash_parser = subparsers.add_parser("compute-hash", help="Compute hash for a request")
    hash_parser.add_argument("--endpoint", required=True, help="Endpoint path (e.g., /v1/chat/completions)")
    hash_parser.add_argument("--method", default="POST", help="HTTP method (default: POST)")
    hash_parser.add_argument("--body", required=True, help="Request body as JSON string")
    hash_parser.add_argument("--test-id", help="Test ID (affects hash for non-model-list endpoints)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "find-hash":
        find_hash(args.hash, args.base_dir, args.test_id)
    elif args.command == "show":
        show_recording(args.file)
    elif args.command == "list-test":
        list_test_recordings(args.test_id, args.base_dir)
    elif args.command == "explain-paths":
        explain_paths(args.test_id, args.base_dir)
    elif args.command == "compute-hash":
        compute_hash(args.endpoint, args.method, args.body, args.test_id)


if __name__ == "__main__":
    main()
