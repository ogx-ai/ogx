# Downloading Models

The `ogx` CLI tool helps you setup and use the OGX. It should be available on your path after installing the `ogx` package.

## Installation

You have two ways to install OGX:

1. **Install as a package**:
   You can install the repository directly from [PyPI](https://pypi.org/project/ogx/) by running the following command:

   ```bash
   pip install ogx
   ```

2. **Install from source**:
   If you prefer to install from the source code, follow these steps:

   ```bash
    mkdir -p ~/local
    cd ~/local
    git clone git@github.com:meta-llama/ogx.git

    uv venv myenv --python 3.12
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate

    cd ogx
    pip install -e .

## Downloading models via Hugging Face CLI

You first need to have models downloaded locally. We recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) to download models.

### Install Hugging Face CLI

First, install the Hugging Face CLI:

```bash
pip install huggingface_hub[cli]
```

### Download models from Hugging Face

You can download models using the `huggingface-cli download` command. Here are some examples:

```bash
# Download Llama 3.2 3B Instruct model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ~/.llama/Llama-3.2-3B-Instruct

# Download Llama 3.2 1B Instruct model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ~/.llama/Llama-3.2-1B-Instruct

# Download Llama Guard 3 1B model
huggingface-cli download meta-llama/Llama-Guard-3-1B --local-dir ~/.llama/Llama-Guard-3-1B

# Download Prompt Guard model
huggingface-cli download meta-llama/Prompt-Guard-86M --local-dir ~/.llama/Prompt-Guard-86M
```

**Important:** You need to authenticate with Hugging Face to download models. You can do this by:

1. Getting your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Running `huggingface-cli login` and entering your token

## List the downloaded models

To list the downloaded models, you can use the Hugging Face CLI:

```bash
# List all downloaded models in your local cache
huggingface-cli scan-cache
```
