---
slug: building-agentic-flows-with-conversations-and-responses
title: "Building a Self-Improving Agent with OGX"
authors: [raghotham]
tags: [agents, responses-api, conversations, prompts, tutorial]
date: 2026-03-01
---

What if your AI agent could improve itself? Most agent tutorials show a single loop — user asks a question, the agent calls some tools, returns an answer. But what happens when you need to systematically improve your agent's behavior over time?

In this post, we build a **ResearchAgent** that answers questions from an internal engineering knowledge base — and gets better at it automatically. The agent uses the Responses API agentic loop with `file_search` and client-side tools to research questions, and it owns its own system prompt. Every N calls, it benchmarks itself by using a different model to judge the results, and rewrites its own prompt via the Prompts API.

This is literally self-referential: **a OGX agent evaluating and improving itself** using the Responses API, Prompts API, and Vector Stores as its toolkit.

<!--truncate-->

## What We're Building

A single `ResearchAgent` class that does two things:

1. **Research** (agentic): Uses the Responses API `while True` loop with server-side `file_search` and client-side function tools (`read_local_file`, `index_document`, `list_local_files`). The agent decides what to search, discovers unindexed local files, reads them, indexes the relevant ones, and searches again with the enriched knowledge base.
2. **Self-improvement** (deterministic): Every N calls to `research()`, the agent runs `evaluate_self()` to benchmark against test cases and `improve_self()` to rewrite its own system prompt. This is a fixed sequence — no LLM-driven tool selection, just the agent measuring and improving its own performance.

```text
┌──────────────────────────────────────────────────────────┐
│  ResearchAgent                                           │
│                                                          │
│  research(question)                                      │
│    Responses API agentic loop (while True):              │
│      Server-side: file_search → Vector Store             │
│      Client-side: read_local_file, index_document,       │
│                   list_local_files                       │
│    Increments call counter; triggers self-improvement    │
│    every N calls                                         │
│                                                          │
│  evaluate_self()                                         │
│    Run all test cases → judge answers (Responses API)    │
│    → log scores (SQLite ledger)                          │
│                                                          │
│  improve_self()                                          │
│    Read feedback → propose new prompt (Responses API)    │
│    → save new version (Prompts API)                      │
└──────────────────────────────────────────────────────────┘
```

## Prerequisites

- [Ollama](https://ollama.com) running locally with two models pulled: `llama3.1:8b` for the research agent and `gpt-oss:20b` as the judge
- A OGX server using the starter distribution, pointed at Ollama via the `OLLAMA_URL` environment variable
- Python SDK: `uv pip install ogx-client`

## The Research Loop

The research agent is the heart of the system — and the showcase for the Responses API agentic pattern. Unlike a simple single-call RAG agent, it has real decisions to make: the vector store might not have enough context, so the agent can discover local files, read them, index the relevant ones, and search again.

It has one server-side tool and three client-side function tools:

- **`file_search`** (server-side): Searches the vector store for relevant documents. The Responses API executes this automatically — no client code needed.
- **`read_local_file(path)`**: Reads an unindexed local file (e.g., a newly written postmortem not yet in the knowledge base).
- **`index_document(file_path)`**: Uploads a file to the vector store via the Files API and `vector_stores.files.create()`. This is the key insight: the agent actively curates the knowledge base.
- **`list_local_files(directory)`**: Discovers available `.md` and `.txt` files in a directory.

The internal `_run_query()` method is the standard Responses API agentic loop — keep calling `responses.create()` until the model stops emitting tool calls:

```python
class ResearchAgent:
    def __init__(self, client, model, vector_store_id, prompt_id, **kwargs):
        self.client = client
        self.model = model
        self.vector_store_id = vector_store_id
        self.prompt_id = prompt_id  # The agent owns its prompt
        self._call_count = 0
        self._tools = {
            "read_local_file": self._read_local_file,
            "index_document": self._index_document,
            "list_local_files": self._list_local_files,
        }
        # Also accepts: judge_model, ledger, test_cases, optimize_every

    def _run_query(self, question, system_prompt):
        """Agentic loop: search, read local files, index, repeat."""
        inputs = question
        tools = self._tool_schemas()

        while True:
            response = self.client.responses.create(
                model=self.model,
                input=inputs,
                instructions=system_prompt,
                tools=tools,
                stream=False,
            )

            # file_search is handled server-side; collect client-side calls
            function_calls = [o for o in response.output if o.type == "function_call"]
            if not function_calls:
                return response.output_text  # Done — no more tool calls

            # Execute each function call and feed results back
            inputs = []
            for fc in function_calls:
                result = self._tools[fc.name](**json.loads(fc.arguments))
                inputs.append(fc)
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": result,
                    }
                )
```

The public `research()` method reads the agent's current prompt, runs the agentic loop, and increments a counter. Every N calls, it triggers self-improvement:

```python
class ResearchAgent:
    ...

    def research(self, question):
        """Answer a question.  Automatically self-improves every N calls."""
        current = self.client.prompts.retrieve(self.prompt_id)
        answer = self._run_query(question, current.prompt)

        self._call_count += 1
        if self.test_cases and self._call_count % self.optimize_every == 0:
            self.evaluate_self()
            self.improve_self()

        return answer
```

In a typical call, the agent searches the vector store via `file_search` (handled server-side). If the retrieved context is insufficient — say, a question about a recent outage whose postmortem hasn't been indexed yet — the agent calls `list_local_files` to discover available documents, `read_local_file` to inspect the relevant one, and `index_document` to add it to the vector store. Then it searches again with the enriched store and writes its final answer.

The `index_document` tool is worth highlighting — it's the agent actively curating its own knowledge base:

```python
class ResearchAgent:
    ...

    def _index_document(self, file_path):
        """Upload a local file to the vector store so it becomes searchable."""
        file = self.client.files.create(
            file=open(file_path, "rb"), purpose="assistants"
        )
        attach = self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id, file_id=file.id
        )
        while attach.status == "in_progress":
            time.sleep(0.5)
            attach = self.client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id, file_id=file.id
            )
        return f"Indexed {file_path} (file_id={file.id}, status={attach.status})"
```

This uses the Files API to upload the document and `vector_stores.files.create()` to attach it to the store. After polling until indexing completes, the file is searchable by `file_search` in subsequent turns of the same query — or in future queries.

## Self-Improvement

The self-improvement cycle is where the agent benchmarks itself, then rewrites its own prompt based on the feedback.

### evaluate_self

`evaluate_self` runs the agent on every test case using its current system prompt, judges each answer with the judge model, and logs scores to the ledger:

```python
class ResearchAgent:
    ...

    def evaluate_self(self):
        """Benchmark against test cases and log scores."""
        current = self.client.prompts.retrieve(self.prompt_id)
        results = []
        for tc in self.test_cases:
            answer = self._run_query(tc["question"], current.prompt)
            judgment = self.client.responses.create(
                model=self.judge_model,
                input=(
                    f"Score the following answer on a scale of 0.0 to 1.0.\n\n"
                    f"Question: {tc['question']}\n"
                    f"Expected: {tc['expected']}\nActual: {answer}\n\n"
                    f'Respond with JSON: {{"score": <float>, "reasoning": "..."}}'
                ),
                stream=False,
            )
            score_data = json.loads(judgment.output_text)
            results.append({**tc, "actual": answer, **score_data})

        avg_score = sum(r["score"] for r in results) / len(results)
        self.ledger.log(self.prompt_id, current.version, avg_score, feedback)
        return {"results": results, "average_score": avg_score, "feedback": feedback}
```

### improve_self

`improve_self` reads the latest evaluation feedback from the ledger and uses the judge model to generate an improved system prompt, then saves it via the Prompts API:

```python
class ResearchAgent:
    ...

    def improve_self(self):
        """Propose and save an improved system prompt."""
        history = self.ledger.history(self.prompt_id)
        latest = history[-1]
        current = self.client.prompts.retrieve(self.prompt_id)

        response = self.client.responses.create(
            model=self.judge_model,
            input=(
                f"Improve this research agent's system prompt based on feedback.\n\n"
                f"Current prompt:\n{current.prompt}\n\n"
                f"Feedback:\n{latest['reasoning']}\n\n"
                f"Return ONLY the improved prompt text."
            ),
            stream=False,
        )
        new_prompt = response.output_text.strip()
        self.client.prompts.update(
            self.prompt_id, prompt=new_prompt, version=current.version
        )
```

The judge model does double duty — scoring answers *and* proposing improvements based on its own feedback. The Prompts API auto-increments versions on each `update()`, and the `version` parameter provides optimistic locking so concurrent experiments don't silently overwrite each other.

### optimize

For initial tuning (before the agent starts serving real queries), `optimize` runs the evaluate/improve cycle in a `for` loop:

```python
class ResearchAgent:
    ...

    def optimize(self, max_iterations=5):
        """Run the evaluate/improve cycle for N iterations."""
        for iteration in range(max_iterations):
            self.evaluate_self()
            self.improve_self()
```

## Running It

First, pull the models and start Ollama, then run the OGX starter distribution pointing at it:

```bash
ollama pull llama3.1:8b
ollama pull gpt-oss:20b
OLLAMA_URL=http://localhost:11434/v1 uv run --with ogx ogx stack run starter
```

The `OLLAMA_URL` environment variable tells the starter distribution to use Ollama as its inference provider. The server starts on `http://localhost:8321` by default.

Then create the agent with some engineering documents. Some docs are indexed in the vector store up front; others live in a local directory for the agent to discover and index on demand:

```python
from ogx_client import OgxClient

client = OgxClient(base_url="http://localhost:8321")

# Create the initial system prompt
initial = client.prompts.create(
    prompt="You are a helpful assistant. Answer questions based on the provided context.",
)

# Create the self-improving research agent
agent = ResearchAgent.from_files(
    client,
    model="ollama/llama3.1:8b",
    name="engineering-kb",
    file_paths=[
        "docs/blog/building-agentic-flows/design/user_service_v2.md",
        "docs/blog/building-agentic-flows/runbooks/deployment_rollback.md",
    ],
    prompt_id=initial.prompt_id,
    local_docs_dir="docs/blog/building-agentic-flows/postmortems",
    judge_model="ollama/gpt-oss:20b",
    ledger=ScoreLedger(),
    test_cases=[
        {
            "question": "What is the deployment rollback procedure?",
            "expected": "Revert the Kubernetes deployment to the previous revision using kubectl rollout undo",
        },
        {
            "question": "What authentication method does the user service use?",
            "expected": "JWT tokens issued by the auth gateway with RS256 signing",
        },
        {
            "question": "What was the root cause of the 2025-02 checkout outage?",
            "expected": "Connection pool exhaustion in the payments service due to missing timeout configuration",
        },
    ],
    optimize_every=10,
)

# Run an initial optimization pass
agent.optimize(max_iterations=5)

# Show the best prompt
result = agent.best_prompt()
print(f"Best prompt (v{result['version']}, score={result['score']:.2f}):")
print(f"  {result['prompt']}")

# Normal usage — the agent self-improves every 10 research() calls
answer = agent.research("What is the deployment rollback procedure?")
print(f"Agent says: {answer}")
```

The full implementation with tool schema generation and all supporting code is available at [self_improving_agent.py](./self_improving_agent.py).

## How It Works Under the Hood

The agent uses *both* kinds of Responses API tools for research:

- **Server-side tools** like `file_search` are executed automatically — the Responses API searches the vector store, retrieves relevant chunks, and feeds them to the model without any client code. This is what makes knowledge base search a single API call.
- **Client-side function tools** (`read_local_file`, `index_document`, `list_local_files`) return tool call objects for you to execute. The `while True` loop dispatches these, and the results feed back into the next `responses.create()` call. This is what lets the agent actively curate its knowledge base.

The agent combines both in a single loop: `file_search` results come back automatically within the response, while function calls need client-side execution. The model sees both sources of information and decides what to do next.

The self-improvement methods don't need any of this machinery. They call `responses.create()` directly for judging and prompt generation — no tool calling, no agentic loop. The Prompts API stores versioned prompt text with optimistic locking, and the SQLite ledger tracks how well each version performed. The `research()` counter ties it all together: the agent serves queries normally, and every N calls it pauses to evaluate and improve itself.

## What's Next

The pattern here — a self-improving agent that benchmarks and rewrites its own prompt — generalizes well beyond research assistants:

- **MCP tools** for connecting to external services (databases, APIs, code execution sandboxes) — the research agent could pull in live data alongside static documents
- **Web search** alongside `file_search` for agents that combine local knowledge with live web results
- **Multiple research agents** with different vector stores, each self-improving independently and specializing in a different knowledge domain

To learn more:

- [Responses API documentation](/docs/building_applications/responses_vs_agents)
- [Conversations API documentation](/docs/api-openai/conformance#conversations)
- [OpenAI API compatibility](/docs/api-openai)
- [Vector Stores documentation](/docs/building_applications/rag)
- [Join our Discord](https://discord.gg/ZAFjsrcw)
