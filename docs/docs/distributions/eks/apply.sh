#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
K8S_DIR="${SCRIPT_DIR}/../k8s"

echo "Setting up AWS EKS-specific storage class..."
kubectl apply -f gp3-topology-aware.yaml

echo "Running main Kubernetes deployment..."
cd "${K8S_DIR}"
./apply.sh "$@"
