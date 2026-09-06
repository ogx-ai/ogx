# Tool Calling Fine-tuning, Inference, and Evaluation with NVIDIA NeMo Microservices and NIM

## Introduction

Tool calling enables Large Language Models (LLMs) to interact with external systems, execute programs, and access real-time information unavailable in their training data. This capability allows LLMs to process natural language queries, map them to specific functions or APIs, and populate required parameters from user inputs. It's essential for building AI agents capable of tasks like checking inventory, retrieving weather data, managing workflows, and more. It imbues generally improved decision making in agents in the presence of real-time information.

### Customizing LLMs for Function Calling

To effectively perform function calling, an LLM must:

- Select the correct function(s)/tool(s) from a set of available options.
- Extract and populate the appropriate parameters for each chosen tool from a user's natural language query.
- In multi-turn (interact with users back-and-forth), and multi-step (break its response into smaller parts) use cases, the LLM may need to plan, and have the capability to chain multiple actions together.

As the number of tools and their complexity increases, customization becomes critical for maintaining accuracy and efficiency. Also, smaller models can achieve comparable performance to larger ones through parameter-efficient techniques like [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685). LoRA is compute- and data-efficient, which involves a smaller one-time investment to train the LoRA adapter, allowing you to reap inference-time benefits with a more efficient "bespoke" model.

### About the xLAM dataset

The Salesforce [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset contains approximately 60,000 training examples specifically designed to enhance language models' function calling capabilities. This dataset has proven particularly valuable for fine-tuning smaller language models (1B-2B parameters) through parameter-efficient techniques like LoRA. The dataset enables models to respond to user queries with executable functions, providing outputs in JSON format that can be directly processed by downstream systems.

### About NVIDIA NeMo Microservices

The NVIDIA NeMo microservices platform provides a flexible foundation for building AI workflows such as fine-tuning, evaluation, running inference, or applying guardrails to AI models on your Kubernetes cluster on-premises or in cloud. Refer to [documentation](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) for further information.

## Objectives

This end-to-end tutorial shows how to leverage the NeMo Microservices platform for customizing [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) using the [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) function-calling dataset, then evaluating its accuracy, and finally safeguarding the customized model behavior.

The following stages will be covered in this set of tutorials:

1. [Preparing Data for fine-tuning and evaluation](./1_data_preparation.ipynb)
2. [Evaluating the accuracy of the customized model](./3_model_evaluation.ipynb)
3. [Adding Guardrails to safeguard your LLM behavior](./4_adding_safety_guardrails.ipynb)

> **Note:** The LoRA fine-tuning of the Llama-3.2-1B-Instruct model takes up to 45 minutes to complete.

## Prerequisites

### Deploy NeMo Microservices

To follow this tutorial, you will need at least two NVIDIA GPUs, which will be allocated as follows:

- **Fine-tuning:** One GPU for fine-tuning the `llama-3.2-1b-instruct` model using NeMo Customizer.
- **Inference:** One GPU for deploying the `llama-3.2-1b-instruct` NIM for inference.

`NOTE`: Notebook [4_adding_safety_guardrails](./4_adding_safety_guardrails.ipynb) asks the user to use one GPU for deploying the `llama-3.1-nemoguard-8b-content-safety` NIM to add content safety guardrails to user input. This will re-use the GPU that was previously used for finetuning in notebook 2.

Refer to the [platform prerequisites and installation guide](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html) to deploy NeMo Microservices.

### Deploy `llama-3.2-1b-instruct` NIM

This step is similar to [NIM deployment instructions](https://docs.nvidia.com/nemo/microservices/latest/get-started/tutorials/deploy-nims.html#deploy-nim-for-llama-3-1-8b-instruct) in documentation, but with the following values:

```bash
# URL to NeMo deployment management service
export NEMO_URL="http://nemo.test"

curl --location "$NEMO_URL/v1/deployment/model-deployments" \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
      "name": "llama-3.2-1b-instruct",
      "namespace": "meta",
      "config": {
         "model": "meta/llama-3.2-1b-instruct",
         "nim_deployment": {
            "distro_name": "nvcr.io/nim/meta/llama-3.2-1b-instruct",
            "image_tag": "1.8.1",
            "pvc_size":   "25Gi",
            "gpu":       1,
            "additional_envs": {
               "NIM_GUIDED_DECODING_BACKEND": "fast_outlines"
            }
         }
      }
   }'
```

The NIM deployment described above should take approximately 10 minutes to go live. You can continue with the remaining steps while the deployment is in progress.

### Managing GPU Resources for Model Deployment (If Applicable)

If you previously deployed the `meta/llama-3.1-8b-instruct` NIM during the [Beginner Tutorial](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html), and are running on a cluster with at most two NVIDIA GPUs, you will need to delete the previous `meta/llama-3.1-8b-instruct` deployment to free up resources. This ensures sufficient GPU availability to run the `meta/llama-3.2-1b-instruct` model while keeping one GPU available for fine-tuning, and another for the content safety NIM.

```bash
export NEMO_URL="http://nemo.test"

curl -X DELETE "$NEMO_URL/v1/deployment/model-deployments/meta/llama-3.1-8b-instruct"
```

### Client-Side Requirements

Ensure you have access to:

1. A Python-enabled machine capable of running Jupyter Lab.
2. Network access to the NeMo Microservices IP and ports.

### Get access to the xLAM dataset

- Go to [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and request access, which should be granted instantly.
- Obtain your [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens).

## Get Started

Navigate to the [data preparation](./1_data_preparation.ipynb) tutorial to get started.

## Other Notes

### About NVIDIA NIM

- The workflow showcased in this tutorial for tool calling fine-tuning is tailored to work with NVIDIA NIM for inference. It won't work with other inference providers (for example, vLLM, SG Lang, TGI).
- For improved inference speeds, we need to use NIM with `fast_outlines` guided decoding system. This is the default if NIM is deployed with the NeMo Microservices Helm Chart. However, if NIM is deployed separately, then users need to set the `NIM_GUIDED_DECODING_BACKEND=fast_outlines` environment variable.

### Limitations with Tool Calling

If you decide to use your own dataset or implement a different data preparation approach:

- There may be a response delay issue in tool calling due to incomplete type info. Tool calls might take over 30 seconds if descriptions for `array` types lack `items` specifications, or if descriptions for `object` types lack `properties` specifications. As a workaround, make sure to include these details (`items` for `array`, `properties` for `object`) in tool descriptions.
- Response Freezing in Tool Calling (Too Many Parameters): Tool calls will freeze the NIM if a tool description includes a function with more than 8 parameters. As a workaround, ensure functions defined in tool descriptions use 8 or fewer parameters. If this does occur, it requires the NIM to be restarted. This will be resolved in the next NIM release.
