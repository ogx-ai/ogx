#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Integration auth tests for OGX
# This script tests authentication and authorization (ABAC) functionality
# Expects token files to be created before running (e.g., by CI workflow or manual setup)

# Function to test API endpoint with authentication
# Usage: test_endpoint <curl_args> <user_token_file> <expected_status> [output_file]
test_endpoint() {
    local curl_args="$1"
    local user_token_file=$2
    local expected_status=$3
    local output_file=${4:-/dev/null}

    local status
    local extra_curl_args=(-s -L -o "$output_file" -w "%{http_code}")

    if [ "$user_token_file" != "none" ]; then
        extra_curl_args+=(-H "Authorization: Bearer $(cat $user_token_file)")
    fi

    set -x
    status=$(curl $curl_args "${extra_curl_args[@]}")
    set +x

    if [ "$status" = "$expected_status" ]; then
        echo "  ✓ Status: $status (expected $expected_status)"
        return 0
    else
        echo "  ✗ Status: $status (expected $expected_status)"
        exit 1
    fi
}

# Check if user tokens exist for ABAC testing
if [ ! -f "ogx-auth-token" ] || [ ! -f "ogx-user1-token" ] || [ ! -f "ogx-user2-token" ]; then
    echo ""
    echo "❌ User tokens not found - expected ogx-user1-token and ogx-user2-token"
    exit 1
fi

echo "Testing /v1/version without token (should succeed)..."
test_endpoint "http://127.0.0.1:8321/v1/version" "none" "200" || exit 1

echo "Testing /v1/providers without token (should fail with 401)..."
test_endpoint "http://127.0.0.1:8321/v1/providers" "none" "401" || exit 1

echo "Testing /v1/providers with valid token (should succeed)..."
test_endpoint "http://127.0.0.1:8321/v1/providers" "ogx-auth-token" "200" "providers.json" || exit 1
cat providers.json | jq . > /dev/null && echo "  ✓ Valid JSON response"

echo ""
echo "Running ABAC tests with user tokens..."

# Create test file
echo "test content" > test-file.txt

echo "Both user1 and user2 can create files..."
test_endpoint "http://127.0.0.1:8321/v1/files -F file=@test-file.txt -F purpose=assistants" "ogx-user1-token" "200" "user1-files.json" || exit 1
test_endpoint "http://127.0.0.1:8321/v1/files -F file=@test-file.txt -F purpose=assistants" "ogx-user2-token" "200" "user2-files.json" || exit 1

echo "user1 can only read their own files..."
test_endpoint "http://127.0.0.1:8321/v1/files" "ogx-user1-token" "200" "user1-files-list.json" || exit 1
USER1_FILE_COUNT=$(jq '.data|length' user1-files-list.json)
echo "User1 has $USER1_FILE_COUNT file(s)"
[ $USER1_FILE_COUNT -eq 1 ] || ( echo "  ✗ User1 should have 1 file, but has $USER1_FILE_COUNT" && exit 1 )
echo "  ✓ User1 can see exactly 1 file"

echo "user2 can read their own file..."
test_endpoint "http://127.0.0.1:8321/v1/files" "ogx-user2-token" "200" "user2-files-list.json" || exit 1
USER2_FILE_COUNT=$(jq '.data|length' user2-files-list.json)
echo "User2 has $USER2_FILE_COUNT file(s)"
[ $USER2_FILE_COUNT -eq 1 ] || ( echo "  ✗ User2 should have 1 file, but has $USER2_FILE_COUNT" && exit 1 )
echo "  ✓ User2 can see their own file"

echo "Both file ids should differ"
FILEID_USER1=$(jq -r '.data[0].id' user1-files-list.json)
FILEID_USER2=$(jq -r '.data[0].id' user2-files-list.json)
[ "$FILEID_USER1" != "$FILEID_USER2" ] || ( echo "  ✗ File IDs should differ" && exit 1 )
echo "  ✓ File IDs differ"

echo "user2 can't delete their own file or other users' files..."
test_endpoint "http://127.0.0.1:8321/v1/files/$FILEID_USER2 -X DELETE" "ogx-user2-token" "404" || exit 1
test_endpoint "http://127.0.0.1:8321/v1/files/$FILEID_USER1 -X DELETE" "ogx-user2-token" "404" || exit 1
echo "  ✓ Delete correctly blocked"

echo "user1 can delete their own files but not other users' files..."
test_endpoint "http://127.0.0.1:8321/v1/files/$FILEID_USER1 -X DELETE" "ogx-user1-token" "200" || exit 1
echo "  ✓ Delete successful"
test_endpoint "http://127.0.0.1:8321/v1/files/$FILEID_USER2 -X DELETE" "ogx-user1-token" "404" || exit 1
echo "  ✓ Delete correctly blocked"

echo "Testing route_policy blocks /v1/models  (DENIED)"
test_endpoint "http://127.0.0.1:8321/v1/models" "ogx-user1-token" "403" || exit 1
echo "  ✓ Wildcard pattern denies /v1/models"

echo ""
echo "✓ ABAC test completed successfully!"
