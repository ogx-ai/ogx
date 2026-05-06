#!/usr/bin/env bash

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Default values
TARGET="stack"
MAX_SECONDS=60
PROMPT_TOKENS=512
OUTPUT_TOKENS=256
RATE_TYPE="concurrent"
RATE="1,2,4,8,16,32,64,128"
STACK_DEPLOYMENT="ogx-benchmark-server"
STACK_URL="http://ogx-benchmark-service:8323/v1/openai"
VLLM_DEPLOYMENT="vllm-server"
OUTPUT_FILE=""

# Parse command line arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --target <stack|vllm>     Target to benchmark (default: stack)"
    echo "  -s, --max-seconds <seconds>   Maximum duration in seconds (default: 60)"
    echo "  -p, --prompt-tokens <tokens>  Number of prompt tokens (default: 512)"
    echo "  -o, --output-tokens <tokens>  Number of output tokens (default: 256)"
    echo "  -r, --rate-type <type>        Rate type (default: concurrent)"
    echo "  -c, --rate                    Rate (default: 1,2,4,8,16,32,64,128)"
    echo "  --output-file <path>          Output file path (default: auto-generated)"
    echo "  --stack-deployment <name>     Name of the stack deployment (default: ogx-benchmark-server)"
    echo "  --vllm-deployment <name>      Name of the vllm deployment (default: vllm-server)"
    echo "  --stack-url <url>             URL of the stack service (default: http://ogx-benchmark-service:8323/v1/openai)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --target vllm                              # Benchmark vLLM direct"
    echo "  $0 --target stack                             # Benchmark OGX (default)"
    echo "  $0 -t vllm -s 60 -p 512 -o 256               # vLLM with custom parameters"
    echo "  $0 --output-file results/my-benchmark.txt     # Specify custom output file"
    echo "  $0 --stack-deployment my-stack-server         # Use custom stack deployment name"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -s|--max-seconds)
            MAX_SECONDS="$2"
            shift 2
            ;;
        -p|--prompt-tokens)
            PROMPT_TOKENS="$2"
            shift 2
            ;;
        -o|--output-tokens)
            OUTPUT_TOKENS="$2"
            shift 2
            ;;
        -r|--rate-type)
            RATE_TYPE="$2"
            shift 2
            ;;
        -c|--rate)
            RATE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --stack-deployment)
            STACK_DEPLOYMENT="$2"
            shift 2
            ;;
        --vllm-deployment)
            VLLM_DEPLOYMENT="$2"
            shift 2
            ;;
        --stack-url)
            STACK_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "stack" && "$TARGET" != "vllm" ]]; then
    echo "Error: Target must be 'stack' or 'vllm'"
    usage
    exit 1
fi

# Set configuration based on target
if [[ "$TARGET" == "vllm" ]]; then
    BASE_URL="http://${VLLM_DEPLOYMENT}:8000"
    JOB_NAME="guidellm-vllm-benchmark-job"
    echo "Benchmarking vLLM direct with GuideLLM..."
else
    BASE_URL="$STACK_URL"
    JOB_NAME="guidellm-stack-benchmark-job"
    echo "Benchmarking OGX with GuideLLM..."
fi


echo "Configuration:"
echo "  Target: $TARGET"
echo "  Base URL: $BASE_URL"
echo "  Max seconds: ${MAX_SECONDS}s"
echo "  Prompt tokens: $PROMPT_TOKENS"
echo "  Output tokens: $OUTPUT_TOKENS"
echo "  Rate type: $RATE_TYPE"
if [[ "$TARGET" == "vllm" ]]; then
    echo "  vLLM deployment: $VLLM_DEPLOYMENT"
else
    echo "  Stack deployment: $STACK_DEPLOYMENT"
fi
echo ""

# Create temporary job yaml
TEMP_YAML="/tmp/guidellm-benchmark-job-temp-$(date +%s).yaml"
cat > "$TEMP_YAML" << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: default
spec:
  template:
    spec:
      containers:
      - name: guidellm-benchmark
        image: python:3.11-slim
        command: ["/bin/bash"]
        args:
        - "-c"
        - |
          # Install uv and guidellm
          pip install uv &&
          uv pip install --system guidellm &&

          # Login to HuggingFace
          uv pip install --system huggingface_hub &&
          python -c "from huggingface_hub import login; login(token='\$HF_TOKEN')" &&

          # Run GuideLLM benchmark and save output
          export COLUMNS=200
          GUIDELLM__PREFERRED_ROUTE="chat_completions" uv run guidellm benchmark run \\
            --target "$BASE_URL" \\
            --rate-type "$RATE_TYPE" \\
            --max-seconds $MAX_SECONDS \\
            --data "prompt_tokens=$PROMPT_TOKENS,output_tokens=$OUTPUT_TOKENS" \\
            --model "$INFERENCE_MODEL" \\
            --rate "$RATE" \\
            --warmup-percent 0.05 \\
            2>&1
        env:
        - name: INFERENCE_MODEL
          value: "meta-llama/Llama-3.2-3B-Instruct"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        resources:
          requests:
            memory: "4Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
      restartPolicy: Never
  backoffLimit: 3
EOF

echo "Cleaning up any existing GuideLLM benchmark job..."
kubectl delete job $JOB_NAME 2>/dev/null || true

echo "Deploying GuideLLM benchmark Job..."
kubectl apply -f "$TEMP_YAML"

echo "Waiting for job to start..."
kubectl wait --for=condition=Ready pod -l job-name=$JOB_NAME --timeout=120s

# Prepare file names and create results directory
mkdir -p results
if [[ -z "$OUTPUT_FILE" ]]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    OUTPUT_FILE="results/guidellm-benchmark-${TARGET}-${TIMESTAMP}.txt"
fi

echo "Following GuideLLM benchmark logs..."
kubectl logs -f job/$JOB_NAME

echo "Job completed. Checking final status..."
kubectl get job $JOB_NAME

# Save benchmark results using kubectl logs
echo "Saving benchmark results..."
kubectl logs job/$JOB_NAME > "$OUTPUT_FILE"

echo "Benchmark output saved to: $OUTPUT_FILE"

# Clean up temporary file
rm -f "$TEMP_YAML"
