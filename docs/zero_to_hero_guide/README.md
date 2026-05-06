# OGX: from Zero to Hero

OGX defines and standardizes the set of core building blocks needed to bring generative AI applications to market. These building blocks are presented in the form of interoperable APIs with a broad set of Providers providing their implementations. These building blocks are assembled into Distributions which are easy for developers to get from zero to production.

This guide will walk you through an end-to-end workflow with OGX with Ollama as the inference provider and ChromaDB as the VectorIO provider. Please note the steps for configuring your provider and distribution will vary depending on the services you use. However, the user experience will remain universal - this is the power of Llama-Stack.

If you're looking for more specific topics, we have a [Zero to Hero Guide](#next-steps) that covers everything from 'Tool Calling' to 'Agents' in detail. Feel free to skip to the end to explore the advanced topics you're interested in.

> If you'd prefer not to set up a local server, explore our notebook on [tool calling with the Together API](Tool_Calling101_Using_Together_Llama_Stack_Server.ipynb). This notebook will show you how to leverage together.ai's OGX Server API, allowing you to get started with OGX without the need for a locally built and running server.

## Table of Contents

- [OGX: from Zero to Hero](#ogx-from-zero-to-hero)
  - [Table of Contents](#table-of-contents)
  - [Setup ollama](#setup-ollama)
  - [Install Dependencies and Set Up Environment](#install-dependencies-and-set-up-environment)
  - [Build, Configure, and Run OGX](#build-configure-and-run-ogx)
  - [Test with `ogx-client` CLI](#test-with-ogx-client-cli)
  - [Test with `curl`](#test-with-curl)
  - [Test with Python](#test-with-python)
    - [1. Create Python Script (`test_ogx.py`)](#1-create-python-script-test_ogxpy)
    - [2. Create a Chat Completion Request in Python](#2-create-a-chat-completion-request-in-python)
    - [3. Run the Python Script](#3-run-the-python-script)
  - [Next Steps](#next-steps)

---

## Setup ollama

1. **Download Ollama App**:
   - Go to [https://ollama.com/download](https://ollama.com/download).
   - Follow instructions based on the OS you are on. For example, if you are on a Mac, download and unzip `Ollama-darwin.zip`.
   - Run the `Ollama` application.

2. **Download the Ollama CLI**:
   Ensure you have the `ollama` command line tool by downloading and installing it from the same website.

3. **Start ollama server**:
   Open the terminal and run:

   ```bash
   ollama serve
   ```

4. **Run the model**:
   Open the terminal and run:

   ```bash
   ollama run llama3.2:3b-instruct-fp16 --keepalive -1m
   ```

   **Note**:
     - The supported models for ogx for now is listed in the [Ollama models file](https://github.com/ogx-ai/ogx/blob/main/ogx/providers/remote/inference/ollama/models.py)
     - `keepalive -1m` is used so that ollama continues to keep the model in memory indefinitely. Otherwise, ollama frees up memory and you would have to run `ollama run` again.

---

## Install Dependencies and Set Up Environment

1. **Install uv**:
   Install [uv](https://docs.astral.sh/uv/) for managing dependencies:

   ```bash
   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install ChromaDB**:
   Install `chromadb` using `uv`:

   ```bash
   uv pip install chromadb
   ```

3. **Run ChromaDB**:
   Start the ChromaDB server:

   ```bash
   chroma run --host localhost --port 8000 --path ./my_chroma_data
   ```

---

## Build, Configure, and Run OGX

1. **Install dependencies**:

   ```bash
   ogx list-deps starter | xargs -L1 uv pip install
   ```

2. **Start the distribution**:

   ```bash
   ogx run starter
   ```

3. **Set the ENV variables by exporting them to the terminal**:

   ```bash
   export OLLAMA_URL="http://localhost:11434"
   export OGX_PORT=8321
   export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
   export SAFETY_MODEL="meta-llama/Llama-Guard-3-1B"
   ```

4. **Run the OGX**:
   Run the stack using uv:

   ```bash
   INFERENCE_MODEL=$INFERENCE_MODEL \
   SAFETY_MODEL=$SAFETY_MODEL \
   OLLAMA_URL=$OLLAMA_URL \
   uv run --with ogx ogx run starter \
      --port $OGX_PORT
   ```

   Note: Every time you run a new model with `ollama run`, you will need to restart the ogx. Otherwise it won't see the new model.

The server will start and listen on `http://localhost:8321`.

---

## Test with `ogx-client` CLI

After setting up the server, open a new terminal window and configure the ogx-client.

1. Configure the CLI to point to the ogx server.

   ```bash
   uv run --with ogx-client ogx-client configure --endpoint http://localhost:8321
   ```

   **Expected Output:**

   ```bash
   Done! You can now use the OGX Client CLI with endpoint http://localhost:8321
   ```

2. Test the CLI by running inference:

   ```bash
   uv run --with ogx-client ogx-client inference chat-completion --message "Write me a 2-sentence poem about the moon"
   ```

   **Expected Output:**

   ```bash
   OpenAIChatCompletion(
      id='chatcmpl-950',
      choices=[
         OpenAIChatCompletionChoice(
               finish_reason='stop',
               index=0,
               message=OpenAIChatCompletionChoiceMessageOpenAIAssistantMessageParam(
                  role='assistant',
                  content='...The moon casts silver threads through the velvet night, a silent bard of shadows, ancient and bright.',
                  name=None,
                  tool_calls=None,
                  refusal=None,
                  annotations=None,
                  audio=None,
                  function_call=None
               ),
               logprobs=None
         )
      ],
      created=1759240813,
      model='meta-llama/Llama-3.2-3B-Instruct',
      object='chat.completion',
      service_tier=None,
      system_fingerprint='fp_ollama',
      usage={
         'completion_tokens': 479,
         'prompt_tokens': 19,
         'total_tokens': 498,
         'completion_tokens_details': None,
         'prompt_tokens_details': None
      },
   )
   ```

## Test with `curl`

After setting up the server, open a new terminal window and verify it's working by sending a `POST` request using `curl`:

```bash
curl http://localhost:$OGX_PORT/v1/chat/completions
-H "Content-Type: application/json"
-d @- <<EOF
{
    "model": "$INFERENCE_MODEL",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2-sentence poem about the moon"}
    ],
      "temperature": 0.7,
      "seed": 42,
      "max_tokens": 512
   }
}
EOF
```

You can check the available models with the command `uv run --with ogx-client ogx-client models list`.

**Expected Output:**

```json
{
    ...
    "content": "... The moon glows softly in the midnight sky,\nA beacon of wonder, as it catches the eye.",
    ...
}
```

---

## Test with Python

You can also interact with the OGX server using a simple Python script. Below is an example:

### 1. Create Python Script (`test_ogx.py`)

```bash
touch test_ogx.py
```

### 2. Create a Chat Completion Request in Python

In `test_ogx.py`, write the following code:

```python
import os
from llama_stack_client import LlamaStackClient

# Get the model ID from the environment variable
INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL")

# Check if the environment variable is se
if INFERENCE_MODEL is None:
    raise ValueError("The environment variable 'INFERENCE_MODEL' is not set.")

# Initialize the client
client = LlamaStackClient(base_url="http://localhost:8321")

# Create a chat completion request
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Write a two-sentence poem about llama."},
    ],
    model=INFERENCE_MODEL,
)

# Print the response
print(response.choices[0].message.content)
```

### 3. Run the Python Script

```bash
uv run --with ogx-client python test_ogx.py
```

**Expected Output:**

```text
The moon glows softly in the midnight sky,
A beacon of wonder, as it catches the eye.
```

With these steps, you should have a functional OGX setup capable of generating text using the specified model. For more detailed information and advanced configurations, refer to some of our documentation below.

This command initializes the model to interact with your local OGX instance.

---

## Next Steps

**Explore Other Guides**: Dive deeper into specific topics by following these guides:

- [Understanding Distribution](https://ogx-ai.github.io/latest/concepts/index.html#distributions)
- [Inference 101](00_Inference101.ipynb)
- [Local and Cloud Model Toggling 101](01_Local_Cloud_Inference101.ipynb)
- [Prompt Engineering](02_Prompt_Engineering101.ipynb)
- [Chat with Image - OGX Vision API](03_Image_Chat101.ipynb)
- [Tool Calling: How to and Details](04_Tool_Calling101.ipynb)
- [Memory API: Show Simple In-Memory Retrieval](05_Memory101.ipynb)
- [Using Safety API in Conversation](06_Safety101.ipynb)
- [Agents API: Explain Components](07_Agents101.ipynb)

**Explore Client SDKs**: Utilize our client SDKs for various languages to integrate OGX into your applications:

- [Python SDK](https://github.com/meta-llama/llama-stack-client-python)
- [Node SDK](https://github.com/ogx-ai/ogx-client-node)
- [Swift SDK](https://github.com/ogx-ai/ogx-client-swift)
- [Kotlin SDK](https://github.com/ogx-ai/ogx-client-kotlin)

**Advanced Configuration**: Learn how to customize your OGX distribution by referring to the [Building a OGX Distribution](https://ogx-ai.github.io/latest/distributions/building_distro.html) guide.

**Explore Example Apps**: Check out [ogx-apps](https://github.com/ogx-ai/ogx-apps/tree/main/examples) for example applications built using OGX.

---
