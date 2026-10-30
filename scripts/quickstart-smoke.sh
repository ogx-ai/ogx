#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Run the docs Quickstart (docs/docs/getting_started/quickstart.mdx) the way a new
# user would: start `ogx go` from an empty directory, outside this repo's uv.lock and
# [tool.uv] settings, then run the documented curl command and the Python snippet
# files the page imports.
#
# Requires a running Ollama on localhost:11434 with llama3.2:3b and nomic-embed-text pulled.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--wheels DIR]

Options:
  --wheels DIR   Start ogx from the ogx and ogx_api wheels in DIR instead of the latest PyPI release
  -h, --help     Show this help

Environment overrides, for debugging against an existing install:
  OGX_GO_CMD       Command that starts the server (default: uvx ogx go)
  SNIPPET_PYTHON   Command that runs the Python snippets (default: uv run --no-project --with openai python)
EOF
}

WHEELS_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheels)
      WHEELS_DIR="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Failed to parse argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNIPPETS_DIR="$REPO_ROOT/docs/docs/getting_started"
RAG_FIXTURE="$REPO_ROOT/tests/integration/responses/fixtures/pdfs/ogx_and_models.pdf"
BASE_URL="https://localhost:8321"
STARTUP_TIMEOUT_SECONDS=600

if [[ -n "${OGX_GO_CMD:-}" ]]; then
  read -r -a ogx_go_cmd <<<"$OGX_GO_CMD"
elif [[ -n "$WHEELS_DIR" ]]; then
  WHEELS_DIR="$(cd "$WHEELS_DIR" && pwd)"
  ogx_wheel=$(ls "$WHEELS_DIR"/ogx-*.whl)
  ogx_api_wheel=$(ls "$WHEELS_DIR"/ogx_api-*.whl)
  ogx_go_cmd=(uvx --from "$ogx_wheel" --with "$ogx_api_wheel" ogx go)
else
  ogx_go_cmd=(uvx ogx go)
fi

if [[ -n "${SNIPPET_PYTHON:-}" ]]; then
  read -r -a snippet_python <<<"$SNIPPET_PYTHON"
else
  snippet_python=(uv run --no-project --with openai python)
fi

WORK_DIR="$(mktemp -d)"
SERVER_LOG="$WORK_DIR/ogx-go.log"
SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ $status -ne 0 && -f "$SERVER_LOG" ]]; then
    echo "=== ogx go output ==="
    cat "$SERVER_LOG"
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

fail() {
  echo "FAILED: $*" >&2
  exit 1
}

step() {
  echo
  echo "=== $* ==="
}

step "Starting server: ${ogx_go_cmd[*]}"
(cd "$WORK_DIR" && exec "${ogx_go_cmd[@]}") >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for ((elapsed = 0; elapsed < STARTUP_TIMEOUT_SECONDS; elapsed++)); do
  if grep -q "Uvicorn running" "$SERVER_LOG"; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "ogx go exited before the server started"
  fi
  sleep 1
done
grep -q "Uvicorn running" "$SERVER_LOG" || fail "ogx go did not start within ${STARTUP_TIMEOUT_SECONDS}s"
grep -E "✓|✗" "$SERVER_LOG" | sed -E 's/ogk_[A-Za-z0-9_-]+/ogk_***/g'

# The quickstart promises Responses and file search, which ogx go only enables once it
# finds an embedding model.
grep -q "Auto-detected embedding model" "$SERVER_LOG" ||
  fail "ogx go did not detect an embedding model, so Responses and file search are disabled"

# The docs tell users to copy one of the keys ogx go prints at startup.
OGX_API_KEY="$(grep -oE 'ogk_[A-Za-z0-9_-]+' "$SERVER_LOG" | head -n 1 || true)"
[[ -n "$OGX_API_KEY" ]] || fail "ogx go did not print an API key"
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  echo "::add-mask::$OGX_API_KEY"
fi
export OGX_API_KEY

step "Verify it works: curl"
models_json="$(curl -k -s -f -H "Authorization: Bearer $OGX_API_KEY" "$BASE_URL/v1/models")" ||
  fail "curl $BASE_URL/v1/models returned an error"
echo "$models_json" | "${snippet_python[@]}" -m json.tool
grep -q '"ollama/llama3.2:3b"' <<<"$models_json" || fail "/v1/models does not list ollama/llama3.2:3b"

cd "$WORK_DIR"

step "Verify it works: quickstart_verify.py"
verify_output="$("${snippet_python[@]}" "$SNIPPETS_DIR/quickstart_verify.py")"
echo "$verify_output"
grep -qE '^ollama/llama3\.2:3b'$'\r''?$' <<<"$verify_output" || fail "quickstart_verify.py did not list ollama/llama3.2:3b"

step "Try it out: quickstart_app.py"
app_output="$("${snippet_python[@]}" "$SNIPPETS_DIR/quickstart_app.py")"
echo "$app_output"
[[ -n "$app_output" ]] || fail "quickstart_app.py printed no response text"

step "Add RAG: quickstart_rag.py"
cp "$RAG_FIXTURE" "$WORK_DIR/my-document.pdf"
rag_output="$("${snippet_python[@]}" "$SNIPPETS_DIR/quickstart_rag.py")"
echo "$rag_output"
[[ -n "$rag_output" ]] || fail "quickstart_rag.py printed no response text"

step "Quickstart passed"
