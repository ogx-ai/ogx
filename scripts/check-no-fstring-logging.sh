#!/bin/bash
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Enforce structlog key-value style logging.
# Use logger.info("msg", key=value) instead of logger.info(f"msg {key}")
# Also block %-style formatting: logger.info("msg %s", val)

# Match all log levels including exception and critical, both quote styles
LOG_METHODS='logger\.\(info\|warning\|error\|debug\|exception\|critical\)'

found=0
for file in "$@"; do
    # Check for f-string logging (double and single quotes)
    if grep -n "${LOG_METHODS}(f[\"']" "$file" | grep -v api_recorder; then
        found=1
    fi
    # Check for %-style formatting (double and single quotes)
    if grep -n "${LOG_METHODS}([\"'].*%[sdrfx]" "$file" | grep -v api_recorder; then
        found=1
    fi
done

if [ "$found" -eq 1 ]; then
    echo ""
    echo "Use structlog key-value style: logger.info(\"msg\", key=value)"
    echo "Do not use f-strings or %-style formatting in log calls."
    exit 1
fi
