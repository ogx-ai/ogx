# llama (server-side) CLI Reference

The `ogx` CLI tool helps you set up and use the OGX. The CLI is available on your path after installing the `ogx` package.

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
   ```

## `ogx` subcommands

1. `stack`: Allows you to build a stack using the `ogx` distribution and run a OGX server. You can read more about how to build a OGX distribution in the [Build your own Distribution](../../distributions/building_distro) documentation.

For downloading models, we recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). See [Downloading models](#downloading-models) for more information.

### Sample Usage

```bash
llama --help
```

```text
usage: llama [-h] {stack} ...

Welcome to the OGX CLI

options:
  -h, --help  show this help message and exit

subcommands:
  {stack}

  stack                 Operations for the OGX / Distributions
```

## Downloading models

You first need to have models downloaded locally. We recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) to download models.

First, install the Hugging Face CLI:

```bash
pip install huggingface_hub[cli]
```

Then authenticate and download models:

```bash
# Authenticate with Hugging Face
huggingface-cli login

# Download a model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ~/.llama/Llama-3.2-3B-Instruct
```

## List the downloaded models

To list the downloaded models, you can use the Hugging Face CLI:

```bash
# List all downloaded models in your local cache
huggingface-cli scan-cache
```
