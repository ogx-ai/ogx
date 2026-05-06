#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

export POSTGRES_USER=ogx
export POSTGRES_DB=ogx
export POSTGRES_PASSWORD=ogx

export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

# HF_TOKEN should be set by the user; base64 encode it for the secret
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN_BASE64=$(echo -n "$HF_TOKEN" | base64)
else
  echo "ERROR: HF_TOKEN not set. You need it for vLLM to download models from Hugging Face."
  exit 1
fi

if [ -z "${GITHUB_CLIENT_ID:-}" ]; then
  echo "ERROR: GITHUB_CLIENT_ID not set. You need it for Github login to work. See the Kubernetes Deployment Guide in the OGX documentation."
  exit 1
fi

if [ -z "${GITHUB_CLIENT_SECRET:-}" ]; then
  echo "ERROR: GITHUB_CLIENT_SECRET not set. You need it for Github login to work. See the Kubernetes Deployment Guide in the OGX documentation."
  exit 1
fi

if [ -z "${OGX_UI_URL:-}" ]; then
  echo "ERROR: OGX_UI_URL not set. Should be set to the external URL of the UI (excluding port). You need it for Github login to work. See the Kubernetes Deployment Guide in the OGX documentation."
  exit 1
fi




set -euo pipefail
set -x

# Apply the HF token secret if HF_TOKEN is provided
if [ -n "${HF_TOKEN:-}" ]; then
  envsubst < ./hf-token-secret.yaml.template | kubectl apply -f -
fi

envsubst < ./vllm-k8s.yaml.template | kubectl apply -f -
envsubst < ./vllm-safety-k8s.yaml.template | kubectl apply -f -
envsubst < ./postgres-k8s.yaml.template | kubectl apply -f -
envsubst < ./chroma-k8s.yaml.template | kubectl apply -f -

kubectl create configmap ogx-config --from-file=stack_run_config.yaml \
  --dry-run=client -o yaml > stack-configmap.yaml

kubectl apply -f stack-configmap.yaml

envsubst < ./stack-k8s.yaml.template | kubectl apply -f -
envsubst < ./ingress-k8s.yaml.template | kubectl apply -f -

envsubst < ./ui-k8s.yaml.template | kubectl apply -f -
