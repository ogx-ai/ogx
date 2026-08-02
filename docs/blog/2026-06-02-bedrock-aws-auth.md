---
slug: ogx-bedrock-aws-auth
title: "Use Amazon Bedrock with OGX Without Managing Bearer Tokens"
authors: [skamenan7]
tags: [ogx, aws, bedrock, sigv4, sts]
date: 2026-06-02
---

OGX now signs Bedrock requests with standard AWS SigV4, so the server uses the same credential chain your platform already runs. No bearer tokens to manage, no custom auth plumbing in your application code.

If your team uses IAM roles, IRSA, or STS for Bedrock access, this means OGX fits into your existing AWS identity model without extra moving parts. Apps talk to one OpenAI-compatible API while OGX handles the provider-specific auth behind the scenes.

For the implementation details, see [issue #4730](https://github.com/ogx-ai/ogx/issues/4730) and [PR #5388](https://github.com/ogx-ai/ogx/pull/5388).

<!--truncate-->

## Why this helps teams adopt OGX

If you are evaluating OGX for production use, this change makes Bedrock easier to fit into an existing AWS environment:

- You can keep using existing OpenAI-compatible clients and agent frameworks.
- You can use standard AWS identity flows, including IAM roles, IRSA, web identity, AWS profiles, and short-lived credentials.
- You do not have to teach every app or service how Bedrock-specific auth works.
- You can keep OGX as the abstraction layer, so switching between Bedrock and other providers stays an infrastructure choice instead of an application rewrite.

The practical effect is less custom auth plumbing at the application layer. Teams can use Bedrock through OGX without giving up the AWS identity flows they already operate.

## What shipped

The core change is a SigV4 path for the Bedrock inference adapter. When `aws_bedrock_bearer_token` is absent, OGX now signs Bedrock requests with AWS SigV4 instead of assuming every request will carry a precomputed bearer token.

That shipped with a few important pieces:

- SigV4 request signing for the Bedrock OpenAI-compatible runtime.
- STS web identity support through `aws_role_arn` and `aws_web_identity_token_file`.
- Automatic refresh of temporary credentials.
- Shared Bedrock config updates so the AWS auth story is consistent across the Bedrock provider code.
- Compatibility with existing bearer-token mode, so this is additive rather than breaking.

For users, the result is simple: if OGX has access to normal AWS credentials, it can talk to Bedrock without requiring a separate token-management workflow.

## How it fits together

From the application side, the client shape stays the same: point a standard client at OGX and let OGX handle the provider details.

```python
import requests
from openai import OpenAI

base_url = "http://localhost:8321"
model_id = requests.get(f"{base_url}/v1/models", timeout=10).json()["data"][0]["id"]

client = OpenAI(base_url=f"{base_url}/v1", api_key="ogx")

response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Explain why OGX is useful"}],
)
```

## Try it now

If you already have AWS credentials available through a profile, IAM role, IRSA, or web identity, this is a short path to a working Bedrock request through OGX.

### 1. Export your AWS environment

```bash
export AWS_DEFAULT_REGION=us-west-2

# Pick the option that matches your environment:
# export AWS_PROFILE=default
# export AWS_ROLE_ARN=arn:aws:iam::<account-id>:role/<role-name>
# For EKS IRSA:
# export AWS_WEB_IDENTITY_TOKEN_FILE=/var/run/secrets/eks.amazonaws.com/serviceaccount/token
```

### 2. Write a minimal `config.yaml`

```bash
cat > config.yaml <<'EOF'
version: 2
distro_name: bedrock-sigv4-demo
apis:
  - inference
  - models
providers:
  inference:
    - provider_id: bedrock-inference
      provider_type: remote::bedrock
      config:
        # aws_bedrock_bearer_token intentionally omitted so OGX uses the AWS credential chain
        region_name: ${env.AWS_DEFAULT_REGION:=us-west-2}
        aws_role_arn: ${env.AWS_ROLE_ARN:=}
        aws_web_identity_token_file: ${env.AWS_WEB_IDENTITY_TOKEN_FILE:=}
        aws_role_session_name: ${env.AWS_ROLE_SESSION_NAME:=ogx-bedrock-demo}
        session_ttl: ${env.AWS_SESSION_TTL:=3600}

storage:
  backends:
    kv_default:
      type: kv_sqlite
      db_path: ./.ogx/kvstore.db
    sql_default:
      type: sql_sqlite
      db_path: ./.ogx/sql_store.db
  stores:
    metadata:
      namespace: registry
      backend: kv_default
    inference:
      table_name: inference_store
      backend: sql_default
      max_write_queue_size: 10000
      num_writers: 4
    prompts:
      namespace: prompts
      backend: kv_default

registered_resources:
  models:
    - metadata: {}
      model_id: openai.gpt-oss-20b-1:0
      provider_id: bedrock-inference
      provider_model_id: openai.gpt-oss-20b-1:0
      model_type: llm
EOF
```

### 3. Start OGX

```bash
uv run ogx run --port 8321 ./config.yaml
```

Leave that terminal running. In a second terminal, use the steps below.

### 4. Verify the model is available

```bash
MODEL_ID=$(curl -s http://localhost:8321/v1/models | jq -r '.data[0].id // empty')
test -n "$MODEL_ID" && echo "Using model: $MODEL_ID"
```

If you want to inspect the full model list:

```bash
curl -s http://localhost:8321/v1/models | jq
```

### 5. Send your first request with `curl`

```bash
curl -s -X POST "http://localhost:8321/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_ID\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Which planet do humans live on?\"}],
    \"stream\": false
  }" | jq -r '.choices[0].message.content'
```

### 6. Send the same request from the OpenAI Python client

```bash
uv run python - <<'PY'
import requests
from openai import OpenAI

base_url = "http://localhost:8321"
model_id = requests.get(f"{base_url}/v1/models", timeout=10).json()["data"][0]["id"]

client = OpenAI(base_url=f"{base_url}/v1", api_key="ogx")

response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Which planet do humans live on?"}],
    stream=False,
)

print(response.choices[0].message.content)
PY
```

If you are only doing a quick local spike and already have a pre-signed Bedrock bearer token, that path still works. For long-running deployments, leave bearer auth unset and let OGX use the standard AWS identity flow instead.

If you do use a bearer token, make sure it was generated for the same AWS Region that OGX is using. A token scoped to `us-east-2` will be rejected by a Bedrock endpoint configured for `us-west-2`, and vice versa.

There are two common paths here: direct SigV4-backed inference for `chat/completions`, and a Bedrock-backed `Responses` path for higher-level workflows such as tool calling. The steps above focus on the shorter `chat/completions` path.

That path has been exercised end to end through the standard OGX surface: model resolution from `/v1/models`, non-streaming and streaming `chat/completions`, fallback to SigV4 when empty bearer overrides are supplied, successful bearer-token overrides when explicitly provided, rejection of invalid bearer overrides, concurrent request isolation, and repeated request smoke checks. For someone evaluating OGX, that is the important part: the AWS-native path works through the same API shape your applications already use.

## Migration

If your config already uses `api_key` or `aws_bearer_token_bedrock`, it still works. No changes required. OGX accepts both the old and new field names through Pydantic aliases.

For new deployments, use the canonical name `aws_bedrock_bearer_token` in your config and `AWS_BEDROCK_BEARER_TOKEN` as the environment variable. Per-request bearer overrides via the `x-ogx-provider-data` header also accept both old and new field names.
