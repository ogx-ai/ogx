# Beginner Fine-tuning, Inference, and Evaluation with NVIDIA NeMo Microservices and NIM

## Introduction

This notebook contains the OGX implementation for an end-to-end workflow for running inference, customizing, and evaluating LLMs using the NVIDIA provider. The NVIDIA provider leverages the NeMo Microservices platform, a collection of microservices that you can use to build AI workflows on your Kubernetes cluster on-prem or in cloud.

### About NVIDIA NeMo Microservices

The NVIDIA NeMo microservices platform provides a flexible foundation for building AI workflows such as fine-tuning, evaluation, running inference, or applying guardrails to AI models on your Kubernetes cluster on-premises or in cloud. Refer to [documentation](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) for further information.

## Objectives

This end-to-end tutorial shows how to leverage the NeMo Microservices platform for customizing Llama-3.1-8B-Instruct using data from the Stanford Question Answering Dataset (SQuAD) reading comprehension dataset, consisting of questions posed on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding passage, or the question is unanswerable.

## Prerequisites

### Deploy NeMo Microservices

Ensure the NeMo Microservices platform is up and running, including the model downloading step for `meta/llama-3.1-8b-instruct`. Please refer to the [installation guide](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-platform/index.html) for instructions.

`NOTE`: The Guardrails step uses the `llama-3.1-nemoguard-8b-content-safety` model to add content safety guardrails to user input. You can either replace this with another model you've already deployed, or deploy this NIM using NeMo Deployment Management Service. This step is similar to [NIM deployment instructions](https://docs.nvidia.com/nemo/microservices/latest/get-started/tutorials/deploy-nims.html#deploy-nim-for-llama-3-1-8b-instruct) in documentation, but with the following values:

```bash
# URL to NeMo deployment management service
export NEMO_URL="http://nemo.test"

curl --location "$NEMO_URL/v1/deployment/model-deployments" \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "name": "llama-3.1-nemoguard-8b-content-safety",
      "namespace": "nvidia",
      "config": {
         "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
         "nim_deployment": {
            "image_name": "nvcr.io/nim/nvidia/llama-3.1-nemoguard-8b-content-safety",
            "image_tag": "1.0.0",
            "pvc_size":   "25Gi",
            "gpu": 1,
            "additional_envs": {
               "NIM_GUIDED_DECODING_BACKEND": "fast_outlines"
            }
         }
      }
   }'
```

The NIM deployment described above should take approximately 10 minutes to go live. You can continue with the remaining steps while the deployment is in progress.

### Client-Side Requirements

Ensure you have access to:

1. A Python-enabled machine capable of running Jupyter Lab.
2. Network access to the NeMo Microservices IP and ports.

## Get Started

Navigate to the [beginner E2E tutorial](./Llama_Stack_NVIDIA_E2E_Flow.ipynb) tutorial to get started.
