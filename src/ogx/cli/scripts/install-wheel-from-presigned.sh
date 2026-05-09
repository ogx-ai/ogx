#!/bin/bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Please provide a URL as an argument."
  exit 1
fi

URL=$1

HEADERS_FILE=$(mktemp)
curl -s -I "$URL" >"$HEADERS_FILE"
FILENAME=$(grep -i "x-manifold-obj-canonicalpath:" "$HEADERS_FILE" | sed -E 's/.*nodes\/[^\/]+\/(.+)/\1/' | tr -d "\r\n")

if [ -z "$FILENAME" ]; then
  echo "Could not find the x-manifold-obj-canonicalpath header."
  echo "HEADERS_FILE contents: "
  cat "$HEADERS_FILE"
  echo ""
  exit 1
fi

echo "Downloading $FILENAME..."

curl -s -L -o "$FILENAME" "$URL"

echo "Installing $FILENAME..."
pip install "$FILENAME"
echo "Successfully installed $FILENAME"

rm -f "$FILENAME"
