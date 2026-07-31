#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

cleanup() {
  # For venv environments, no special cleanup is needed
  # This function exists to avoid "function not found" errors
  local env_name="$1"
  echo "Cleanup called for environment: $env_name"
}

handle_int() {
  if [ -n "$ENVNAME" ]; then
    cleanup "$ENVNAME"
  fi
  exit 1
}

handle_exit() {
  if [ $? -ne 0 ]; then
    echo -e "\033[1;31mABORTING.\033[0m"
    if [ -n "$ENVNAME" ]; then
      cleanup "$ENVNAME"
    fi
  fi
}



# check if a command is present
is_command_available() {
  command -v "$1" &>/dev/null
}
