# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Self-improving research agent using OGX APIs.

The ResearchAgent answers questions from an internal engineering knowledge base
using the Responses API agentic loop (server-side ``file_search`` + client-side
function tools).  It also owns its own system prompt and periodically
self-improves: after every N ``research()`` calls it benchmarks itself against
test cases, judges the results with a second model, and rewrites its own system
prompt via the Prompts API.
"""

import inspect
import json
import os
import sqlite3
import time
from typing import Annotated, get_args, get_origin

from llama_stack_client import LlamaStackClient


# ---------------------------------------------------------------------------
# Tool schema generation — derives OpenAI function tool definitions from
# Python function signatures using type hints and Annotated descriptions.
# ---------------------------------------------------------------------------

PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def fn_to_tool_schema(fn) -> dict:
    """Derive an OpenAI function tool schema from a Python function."""
    sig = inspect.signature(fn)
    hints = fn.__annotations__
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        hint = hints.get(name, str)

        # Extract description from Annotated[type, "description"]
        description = None
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            hint = args[0]
            if len(args) > 1 and isinstance(args[1], str):
                description = args[1]

        # Handle Optional (T | None)
        is_optional = False
        if get_origin(hint) is type(int | None):  # UnionType
            inner_types = [t for t in get_args(hint) if t is not type(None)]
            if inner_types:
                hint = inner_types[0]
                is_optional = True

        prop = {"type": PYTHON_TYPE_TO_JSON.get(hint, "string")}
        if description:
            prop["description"] = description
        properties[name] = prop

        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    return {
        "type": "function",
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# ScoreLedger — SQLite-backed record of (prompt_version, score) pairs.
# ---------------------------------------------------------------------------


class ScoreLedger:
    def __init__(self, db_path: str = "prompt_optimizer.db"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                prompt_id TEXT,
                version INTEGER,
                score REAL,
                reasoning TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def log(self, prompt_id: str, version: int, score: float, reasoning: str):
        self.db.execute(
            "INSERT INTO results (prompt_id, version, score, reasoning) VALUES (?, ?, ?, ?)",
            (prompt_id, version, score, reasoning),
        )
        self.db.commit()

    def history(self, prompt_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT version, score, reasoning, timestamp FROM results WHERE prompt_id = ? ORDER BY version",
            (prompt_id,),
        ).fetchall()
        return [
            {"version": r[0], "score": r[1], "reasoning": r[2], "timestamp": r[3]}
            for r in rows
        ]


# ---------------------------------------------------------------------------
# ResearchAgent — a self-improving agentic research assistant.
#
# Uses the Responses API ``while True`` loop with server-side file_search and
# client-side function tools to answer questions.  Owns its system prompt and
# periodically evaluates + improves itself.
# ---------------------------------------------------------------------------


class ResearchAgent:
    def __init__(
        self,
        client: LlamaStackClient,
        model: str,
        vector_store_id: str,
        prompt_id: str,
        local_docs_dir: str | None = None,
        judge_model: str | None = None,
        ledger: ScoreLedger | None = None,
        test_cases: list[dict] | None = None,
        optimize_every: int = 10,
    ):
        self.client = client
        self.model = model
        self.vector_store_id = vector_store_id
        self.prompt_id = prompt_id
        self.local_docs_dir = local_docs_dir
        self.judge_model = judge_model
        self.ledger = ledger
        self.test_cases = test_cases or []
        self.optimize_every = optimize_every
        self._call_count = 0

        # Client-side function tools the agent can call
        self._tools = {
            "read_local_file": self._read_local_file,
            "index_document": self._index_document,
            "list_local_files": self._list_local_files,
        }

    @classmethod
    def from_files(
        cls,
        client: LlamaStackClient,
        model: str,
        name: str,
        file_paths: list[str],
        prompt_id: str,
        local_docs_dir: str | None = None,
        judge_model: str | None = None,
        ledger: ScoreLedger | None = None,
        test_cases: list[dict] | None = None,
        optimize_every: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ) -> "ResearchAgent":
        """Create a ResearchAgent with a new vector store populated from files."""
        vector_store = client.vector_stores.create(
            name=name,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

        for path in file_paths:
            file = client.files.create(file=open(path, "rb"), purpose="assistants")
            attach = client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file.id,
            )
            while attach.status == "in_progress":
                time.sleep(0.5)
                attach = client.vector_stores.files.retrieve(
                    vector_store_id=vector_store.id, file_id=file.id,
                )

        return cls(
            client,
            model,
            vector_store.id,
            prompt_id,
            local_docs_dir=local_docs_dir,
            judge_model=judge_model,
            ledger=ledger,
            test_cases=test_cases,
            optimize_every=optimize_every,
        )

    # -- Client-side function tools ------------------------------------------

    @staticmethod
    def _read_local_file(
        path: Annotated[str, "Path to the local file to read"],
    ) -> str:
        """Read an unindexed local file and return its contents."""
        with open(path) as f:
            return f.read()

    def _index_document(
        self,
        file_path: Annotated[str, "Path to the local file to index"],
    ) -> str:
        """Upload a local file to the vector store so it becomes searchable."""
        file = self.client.files.create(
            file=open(file_path, "rb"), purpose="assistants",
        )
        attach = self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id, file_id=file.id,
        )
        while attach.status == "in_progress":
            time.sleep(0.5)
            attach = self.client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id, file_id=file.id,
            )
        return f"Indexed {file_path} (file_id={file.id}, status={attach.status})"

    @staticmethod
    def _list_local_files(
        directory: Annotated[str, "Directory to list files from"],
    ) -> str:
        """List .md and .txt files in a directory that could be indexed."""
        files = [
            os.path.join(directory, f)
            for f in sorted(os.listdir(directory))
            if f.endswith((".md", ".txt"))
        ]
        return json.dumps(files)

    # -- Agentic research loop -----------------------------------------------

    def _tool_schemas(self) -> list[dict]:
        """Return tool schemas for file_search + client-side function tools."""
        return [
            {"type": "file_search", "vector_store_ids": [self.vector_store_id]},
            *[fn_to_tool_schema(fn) for fn in self._tools.values()],
        ]

    def _run_query(self, question: str, system_prompt: str) -> str:
        """Raw agentic loop — search, read local files, index, repeat."""
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

            # Collect any function calls (file_search is handled server-side)
            function_calls = [o for o in response.output if o.type == "function_call"]
            if not function_calls:
                return response.output_text

            # Execute each function call and feed results back
            inputs = []
            for fc in function_calls:
                result = self._tools[fc.name](**json.loads(fc.arguments))
                inputs.append(fc)
                inputs.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": result if isinstance(result, str) else json.dumps(result),
                })

    def research(self, question: str) -> str:
        """Answer a question.  Automatically self-improves every N calls."""
        current = self.client.prompts.retrieve(self.prompt_id)
        answer = self._run_query(question, current.prompt)

        self._call_count += 1
        if (
            self.judge_model
            and self.test_cases
            and self._call_count % self.optimize_every == 0
        ):
            print(f"\n[auto-optimize] {self._call_count} calls — running self-improvement")
            self.evaluate_self()
            self.improve_self()

        return answer

    # -- Self-improvement ----------------------------------------------------

    def evaluate_self(self) -> dict:
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
                    f"Expected answer: {tc['expected']}\n"
                    f"Actual answer: {answer}\n\n"
                    f'Respond with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}'
                ),
                stream=False,
            )
            score_data = json.loads(judgment.output_text)
            results.append({
                "question": tc["question"],
                "expected": tc["expected"],
                "actual": answer,
                "score": score_data["score"],
                "reasoning": score_data["reasoning"],
            })

        avg_score = sum(r["score"] for r in results) / len(results)
        feedback = "; ".join(
            f"Q: {r['question'][:40]}… → {r['score']:.1f} ({r['reasoning']})"
            for r in results
        )
        self.ledger.log(self.prompt_id, current.version, avg_score, feedback)
        print(f"  evaluate_self: v{current.version} scored {avg_score:.2f}")
        return {"results": results, "average_score": avg_score, "feedback": feedback}

    def improve_self(self) -> dict:
        """Propose and save an improved system prompt based on the latest evaluation."""
        history = self.ledger.history(self.prompt_id)
        latest = history[-1]
        current = self.client.prompts.retrieve(self.prompt_id)

        response = self.client.responses.create(
            model=self.judge_model,
            input=(
                f"You are improving a research agent's system prompt based on evaluation feedback.\n\n"
                f"Current prompt:\n{current.prompt}\n\n"
                f"Judge feedback:\n{latest['reasoning']}\n\n"
                f"Write an improved system prompt that addresses the feedback. "
                f"Return ONLY the new prompt text, nothing else."
            ),
            stream=False,
        )
        new_prompt = response.output_text.strip()
        updated = self.client.prompts.update(
            self.prompt_id, prompt=new_prompt, version=current.version,
        )
        print(f"  improve_self: saved v{updated.version}")
        return {"prompt": new_prompt, "version": updated.version}

    def optimize(self, max_iterations: int = 5):
        """Run the evaluate/improve cycle for N iterations."""
        for iteration in range(max_iterations):
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'=' * 60}")
            self.evaluate_self()
            self.improve_self()

    def best_prompt(self) -> dict:
        """Return the highest-scoring prompt from the ledger."""
        history = self.ledger.history(self.prompt_id)
        best = max(history, key=lambda h: h["score"])
        prompt = self.client.prompts.retrieve(self.prompt_id, version=best["version"])
        return {"version": best["version"], "score": best["score"], "prompt": prompt.prompt}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = LlamaStackClient(base_url="http://localhost:8321")

    MODEL = "ollama/llama3.1:8b"
    JUDGE_MODEL = "ollama/gpt-oss:20b"

    TEST_CASES = [
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
    ]

    # Create the initial system prompt
    initial = client.prompts.create(
        prompt="You are a helpful assistant. Answer questions based on the provided context.",
    )

    # Create the research agent — some docs are indexed up front,
    # others live in a local directory for the agent to discover.
    agent = ResearchAgent.from_files(
        client,
        model=MODEL,
        name="engineering-kb",
        file_paths=[
            "docs/blog/building-agentic-flows/design/user_service_v2.md",
            "docs/blog/building-agentic-flows/runbooks/deployment_rollback.md",
        ],
        prompt_id=initial.prompt_id,
        local_docs_dir="docs/blog/building-agentic-flows/postmortems",
        judge_model=JUDGE_MODEL,
        ledger=ScoreLedger(),
        test_cases=TEST_CASES,
        optimize_every=10,
    )

    # Run an initial optimization pass (5 evaluate/improve cycles)
    agent.optimize(max_iterations=5)

    # Show results
    print(f"\n{'=' * 60}")
    print("Optimization complete!")
    print(f"{'=' * 60}")
    for h in agent.ledger.history(initial.prompt_id):
        print(f"  v{h['version']}: score={h['score']:.2f} — {h['reasoning'][:80]}")

    result = agent.best_prompt()
    print(f"\nBest prompt (v{result['version']}, score={result['score']:.2f}):")
    print(f"  {result['prompt']}")

    # Normal usage — the agent self-improves every 10 research() calls
    answer = agent.research("What is the deployment rollback procedure?")
    print(f"\nAgent says: {answer}")
