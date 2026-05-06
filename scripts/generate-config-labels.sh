#!/usr/bin/env bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Generate OCI label flags for distribution configs
set -euo pipefail

DISTRO_NAME="${1:-}"
VERSION="${2:-}"
DEFAULT_CONFIG="${3:-config.yaml}"

if [ -z "$DISTRO_NAME" ]; then
    echo "Usage: $0 DISTRO_NAME VERSION [DEFAULT_CONFIG]" >&2
    exit 1
fi

if [ -z "$VERSION" ]; then
    echo "Usage: $0 DISTRO_NAME VERSION [DEFAULT_CONFIG]" >&2
    exit 1
fi

DISTRO_DIR="src/ogx/distributions/${DISTRO_NAME}"

if [ ! -d "$DISTRO_DIR" ]; then
    echo "Error: Distribution directory not found: $DISTRO_DIR" >&2
    exit 1
fi

# Validate default config exists before processing
if [ ! -f "$DISTRO_DIR/$DEFAULT_CONFIG" ]; then
    echo "ERROR: Default config '$DEFAULT_CONFIG' not found in $DISTRO_DIR" >&2
    exit 1
fi

CONFIG_LIST=""

# Process each YAML file (excluding build.yaml)
for yaml_file in "$DISTRO_DIR"/*.yaml; do
    [ -f "$yaml_file" ] || continue
    filename=$(basename "$yaml_file")
    [ "$filename" = "build.yaml" ] && continue

    # Base64-encode the content (handle macOS/Linux differences)
    encoded=$(base64 -w 0 < "$yaml_file" 2>/dev/null || base64 < "$yaml_file" | tr -d '\n')
    encoded_size=$(echo -n "$encoded" | wc -c)

    # Output label flag (one per line for safe array construction)
    echo "--label"
    echo "com.ogx.config.${filename}=${encoded}"

    # Build config list
    if [ -z "$CONFIG_LIST" ]; then
        CONFIG_LIST="$filename"
    else
        CONFIG_LIST="${CONFIG_LIST},${filename}"
    fi

    echo "Generated label for: $filename (${encoded_size} bytes)" >&2
done


# Output metadata labels (one per line)
echo "--label"
echo "com.ogx.distribution.name=${DISTRO_NAME}"
echo "--label"
echo "com.ogx.distribution.version=${VERSION}"
echo "--label"
echo "com.ogx.distribution.default-config=${DEFAULT_CONFIG}"
echo "--label"
echo "com.ogx.distribution.configs=${CONFIG_LIST}"

# Output OCI standard labels (one per line)
echo "--label"
echo "org.opencontainers.image.title=OGX - ${DISTRO_NAME}"
echo "--label"
echo "org.opencontainers.image.version=${VERSION}"
