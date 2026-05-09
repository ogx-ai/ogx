# llama (client-side) CLI Reference

The `ogx-client` CLI allows you to query information about the distribution.

## Basic Commands

### `ogx-client`

```bash
ogx-client
Usage: ogx-client [OPTIONS] COMMAND [ARGS]...

  Welcome to the ogx-client CLI - a command-line interface for
  interacting with OGX

Options:
  --version        Show the version and exit.
  --endpoint TEXT  OGX distribution endpoint
  --api-key TEXT   OGX distribution API key
  --config TEXT    Path to config file
  --help           Show this message and exit.

Commands:
  configure          Configure OGX Client CLI.
  datasets           Manage datasets.
  eval               Run evaluation tasks.
  eval_tasks         Manage evaluation tasks.
  inference          Inference (chat).
  inspect            Inspect server configuration.
  models             Manage GenAI models.
  post_training      Post-training.
  providers          Manage API providers.
  scoring_functions  Manage scoring functions.
  shields            Manage safety shield services.
  toolgroups         Manage available tool groups.
```

### `ogx-client configure`

Configure OGX Client CLI.

```bash
ogx-client configure
> Enter the host name of the OGX distribution server: localhost
> Enter the port number of the OGX distribution server: 8321
Done! You can now use the OGX Client CLI with endpoint http://localhost:8321
```

Optional arguments:

- `--endpoint`: OGX distribution endpoint
- `--api-key`: OGX distribution API key

## `ogx-client inspect version`

Inspect server configuration.

```bash
ogx-client inspect version
```

```bash
VersionInfo(version='0.2.14')
```

### `ogx-client providers list`

Show available providers on distribution endpoint

```bash
ogx-client providers list
```

```text
+-----------+----------------+-----------------+
| API       | Provider ID    | Provider Type   |
+===========+================+=================+
| scoring   | meta0          | builtin  |
+-----------+----------------+-----------------+
| datasetio | meta0          | builtin  |
+-----------+----------------+-----------------+
| inference | vllm0          | remote::vllm    |
+-----------+----------------+-----------------+
| memory    | builtin | builtin  |
+-----------+----------------+-----------------+
| agents    | builtin | builtin  |
+-----------+----------------+-----------------+
| safety    | builtin | builtin  |
+-----------+----------------+-----------------+
```

### `ogx-client providers inspect`

Show specific provider configuration on distribution endpoint

```bash
ogx-client providers inspect <provider_id>
```

## Inference

Inference (chat).

### `ogx-client inference chat-completion`

Show available inference chat completion endpoints on distribution endpoint

```bash
ogx-client inference chat-completion --message <message> [--stream] [--session] [--model-id]
```

```bash
OpenAIChatCompletion(
    id='chatcmpl-aacd11f3-8899-4ec5-ac5b-e655132f6891',
    choices=[
        OpenAIChatCompletionChoice(
            finish_reason='stop',
            index=0,
            message=OpenAIChatCompletionChoiceMessageOpenAIAssistantMessageParam(
                role='assistant',
                content='The captain of the whaleship Pequod in Nathaniel Hawthorne\'s novel "Moby-Dick" is Captain
Ahab. He\'s a vengeful and obsessive old sailor who\'s determined to hunt down and kill the white sperm whale
Moby-Dick, whom he\'s lost his leg to in a previous encounter.',
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
    created=1752578797,
    model='llama3.2:3b-instruct-fp16',
    object='chat.completion',
    service_tier=None,
    system_fingerprint='fp_ollama',
    usage={
        'completion_tokens': 67,
        'prompt_tokens': 33,
        'total_tokens': 100,
        'completion_tokens_details': None,
        'prompt_tokens_details': None
    }
)
```

Required arguments:
**Note:** At least one of these parameters is required for chat completion

- `--message`: Message
- `--session`: Start a Chat Session

Optional arguments:

- `--stream`: Stream
- `--model-id`: Model ID

## Model Management

Manage GenAI models.

### `ogx-client models list`

Show available llama models at distribution endpoint

```bash
ogx-client models list
```

```text
Available Models

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ model_type   ┃ identifier                           ┃ provider_resource_id         ┃ metadata  ┃ provider_id ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ llm          │ meta-llama/Llama-3.2-3B-Instruct     │ llama3.2:3b-instruct-fp16    │           │ ollama      │
└──────────────┴──────────────────────────────────────┴──────────────────────────────┴───────────┴─────────────┘

Total models: 1
```

### `ogx-client models get`

Show details of a specific model at the distribution endpoint

```bash
ogx-client models get Llama3.1-8B-Instruct
```

```text
+----------------------+----------------------+----------------------------------------------------------+---------------+
| identifier           | llama_model          | metadata                                                 | provider_id   |
+======================+======================+==========================================================+===============+
| Llama3.1-8B-Instruct | Llama3.1-8B-Instruct | {'huggingface_repo': 'meta-llama/Llama-3.1-8B-Instruct'} | vllm0         |
+----------------------+----------------------+----------------------------------------------------------+---------------+
```

```bash
ogx-client models get Random-Model

Model RandomModel is not found at distribution endpoint host:port. Please ensure endpoint is serving specified model.
```

### `ogx-client models register`

Register a new model at distribution endpoint

```bash
ogx-client models register <model_id> [--provider-id <provider_id>] [--provider-model-id <provider_model_id>] [--metadata <metadata>] [--model-type <model_type>]
```

Required arguments:

- `MODEL_ID`: Model ID
- `--provider-id`: Provider ID for the model

Optional arguments:

- `--provider-model-id`: Provider's model ID
- `--metadata`: JSON metadata for the model
- `--model-type`: Model type: `llm`, `embedding`

### `ogx-client models unregister`

Unregister a model from distribution endpoint

```bash
ogx-client models unregister <model_id>
```

## Shield Management

Manage safety shield services.

### `ogx-client shields list`

Show available safety shields on distribution endpoint

```bash
ogx-client shields list
```

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ identifier                       ┃ provider_alias                                                        ┃ params                ┃ provider_id                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ollama                           │ ollama/llama-guard3:1b                                                │                       │ llama-guard                        │
└──────────────────────────────────┴───────────────────────────────────────────────────────────────────────┴───────────────────────┴────────────────────────────────────┘
```

### `ogx-client shields register`

Register a new safety shield

```bash
ogx-client shields register --shield-id <shield-id> [--provider-id <provider-id>] [--provider-shield-id <provider-shield-id>] [--params <params>]
```

Required arguments:

- `--shield-id`: ID of the shield

Optional arguments:

- `--provider-id`: Provider ID for the shield
- `--provider-shield-id`: Provider's shield ID
- `--params`: JSON configuration parameters for the shield

## Eval execution

Run evaluation tasks.

### `ogx-client eval run-benchmark`

Run a evaluation benchmark task

```bash
ogx-client eval run-benchmark <eval-task-id1> [<eval-task-id2> ...] --eval-task-config <config-file> --output-dir <output-dir> --model-id <model-id> [--num-examples <num>] [--visualize] [--repeat-penalty <repeat-penalty>] [--top-p <top-p>] [--max-tokens <max-tokens>]
```

Required arguments:

- `--eval-task-config`: Path to the eval task config file in JSON format
- `--output-dir`: Path to the directory where evaluation results will be saved
- `--model-id`: model id to run the benchmark eval on

Optional arguments:

- `--num-examples`: Number of examples to evaluate (useful for debugging)
- `--visualize`: If set, visualizes evaluation results after completion
- `--repeat-penalty`: repeat-penalty in the sampling params to run generation
- `--top-p`: top-p in the sampling params to run generation
- `--max-tokens`: max-tokens in the sampling params to run generation
- `--temperature`: temperature in the sampling params to run generation

Example benchmark_config.json:

```json
{
    "type": "benchmark",
    "eval_candidate": {
        "type": "model",
        "model": "Llama3.1-405B-Instruct",
        "sampling_params": {
            "strategy": "greedy",
        }
    }
}
```

### `ogx-client eval run-scoring`

Run scoring from application datasets

```bash
ogx-client eval run-scoring <eval-task-id> --output-dir <output-dir> [--num-examples <num>] [--visualize]
```

Required arguments:

- `--output-dir`: Path to the directory where scoring results will be saved

Optional arguments:

- `--num-examples`: Number of examples to evaluate (useful for debugging)
- `--visualize`: If set, visualizes scoring results after completion
- `--scoring-params-config`: Path to the scoring params config file in JSON format
- `--dataset-id`: Pre-registered dataset_id to score (from ogx-client datasets list)
- `--dataset-path`: Path to the dataset file to score

## Eval Tasks

Manage evaluation tasks.

### `ogx-client eval_tasks list`

Show available eval tasks on distribution endpoint

```bash
ogx-client eval_tasks list
```

### `ogx-client eval_tasks register`

Register a new eval task

```bash
ogx-client eval_tasks register --eval-task-id <eval-task-id> --dataset-id <dataset-id> --scoring-functions <scoring-functions> [--provider-id <provider-id>] [--provider-eval-task-id <provider-eval-task-id>] [--metadata <metadata>]
```

Required arguments:

- `--eval-task-id`: ID of the eval task
- `--dataset-id`: ID of the dataset to evaluate
- `--scoring-functions`: Scoring functions to use for evaluation

Optional arguments:

- `--provider-id`: Provider ID for the eval task
- `--provider-eval-task-id`: Provider's eval task ID

## Tool Group Management

Manage available tool groups.

### `ogx-client toolgroups list`

Show available llama toolgroups at distribution endpoint

```bash
ogx-client toolgroups list
```

```text
+---------------------------+------------------+------+---------------+
| identifier                | provider_id      | args | mcp_endpoint  |
+===========================+==================+======+===============+
| builtin::file_search              | file-search      | None | None          |
+---------------------------+------------------+------+---------------+
| builtin::websearch        | tavily-search    | None | None          |
+---------------------------+------------------+------+---------------+
```

### `ogx-client toolgroups get`

Get available llama toolgroups by id

```bash
ogx-client toolgroups get <toolgroup_id>
```

Shows detailed information about a specific toolgroup. If the toolgroup is not found, displays an error message.

Required arguments:

- `TOOLGROUP_ID`: ID of the tool group

### `ogx-client toolgroups register`

Register a new toolgroup at distribution endpoint

```bash
ogx-client toolgroups register <toolgroup_id> [--provider-id <provider-id>] [--provider-toolgroup-id <provider-toolgroup-id>] [--mcp-config <mcp-config>] [--args <args>]
```

Required arguments:

- `TOOLGROUP_ID`: ID of the tool group

Optional arguments:

- `--provider-id`: Provider ID for the toolgroup
- `--provider-toolgroup-id`: Provider's toolgroup ID
- `--mcp-config`: JSON configuration for the MCP endpoint
- `--args`: JSON arguments for the toolgroup

### `ogx-client toolgroups unregister`

Unregister a toolgroup from distribution endpoint

```bash
ogx-client toolgroups unregister <toolgroup_id>
```

Required arguments:

- `TOOLGROUP_ID`: ID of the tool group

## Datasets Management

Manage datasets.

### `ogx-client datasets list`

Show available datasets on distribution endpoint

```bash
ogx-client datasets list
```

### `ogx-client datasets register`

```bash
ogx-client datasets register --dataset_id <dataset_id> --purpose <purpose> [--url <url] [--dataset-path <dataset-path>] [--dataset-id <dataset-id>] [--metadata <metadata>]
```

Required arguments:

- `--dataset_id`: Id of the dataset
- `--purpose`: Purpose of the dataset

Optional arguments:

- `--metadata`: Metadata of the dataset
- `--url`: URL of the dataset
- `--dataset-path`: Local file path to the dataset. If specified, upload dataset via URL

### `ogx-client datasets unregister`

Remove a dataset

```bash
ogx-client datasets unregister <dataset-id>
```

Required arguments:

- `DATASET_ID`: Id of the dataset

## Scoring Functions Management

Manage scoring functions.

### `ogx-client scoring_functions list`

Show available scoring functions on distribution endpoint

```bash
ogx-client scoring_functions list
```

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ identifier                                 ┃ provider_id  ┃ description                                                   ┃ type             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ basic::docvqa                              │ basic        │ DocVQA Visual Question & Answer scoring function              │ scoring_function │
│ basic::equality                            │ basic        │ Returns 1.0 if the input is equal to the target, 0.0          │ scoring_function │
│                                            │              │ otherwise.                                                    │                  │
└────────────────────────────────────────────┴──────────────┴───────────────────────────────────────────────────────────────┴──────────────────┘
```

### `ogx-client scoring_functions register`

Register a new scoring function

```bash
ogx-client scoring_functions register --scoring-fn-id <scoring-fn-id> --description <description> --return-type <return-type> [--provider-id <provider-id>] [--provider-scoring-fn-id <provider-scoring-fn-id>] [--params <params>]
```

Required arguments:

- `--scoring-fn-id`: Id of the scoring function
- `--description`: Description of the scoring function
- `--return-type`: Return type of the scoring function

Optional arguments:

- `--provider-id`: Provider ID for the scoring function
- `--provider-scoring-fn-id`: Provider's scoring function ID
- `--params`: Parameters for the scoring function in JSON format

## Post Training Management

Post-training.

### `ogx-client post_training list`

Show the list of available post training jobs

```bash
ogx-client post_training list
```

```bash
["job-1", "job-2", "job-3"]
```

### `ogx-client post_training artifacts`

Get the training artifacts of a specific post training job

```bash
ogx-client post_training artifacts --job-uuid <job-uuid>
```

```bash
JobArtifactsResponse(checkpoints=[], job_uuid='job-1')
```

Required arguments:

- `--job-uuid`: Job UUID

### `ogx-client post_training supervised_fine_tune`

Kick off a supervised fine tune job

```bash
ogx-client post_training supervised_fine_tune --job-uuid <job-uuid> --model <model> --algorithm-config <algorithm-config> --training-config <training-config> [--checkpoint-dir <checkpoint-dir>]
```

Required arguments:

- `--job-uuid`: Job UUID
- `--model`: Model ID
- `--algorithm-config`: Algorithm Config
- `--training-config`: Training Config

Optional arguments:

- `--checkpoint-dir`: Checkpoint Config

### `ogx-client post_training status`

Show the status of a specific post training job

```bash
ogx-client post_training status --job-uuid <job-uuid>
```

```bash
JobStatusResponse(
    checkpoints=[],
    job_uuid='job-1',
    status='completed',
    completed_at="",
    resources_allocated="",
    scheduled_at="",
    started_at=""
)
```

Required arguments:

- `--job-uuid`: Job UUID

### `ogx-client post_training cancel`

Cancel the training job

```bash
ogx-client post_training cancel --job-uuid <job-uuid>
```

```bash
# This functionality is not yet implemented for ogx-client
╭────────────────────────────────────────────────────────────╮
│ Failed to post_training cancel_training_job                │
│                                                            │
│ Error Type: InternalServerError                            │
│ Details: Error code: 501 - {'detail': 'Not implemented: '} │
╰────────────────────────────────────────────────────────────╯
```

Required arguments:

- `--job-uuid`: Job UUID
