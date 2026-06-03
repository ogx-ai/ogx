# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Abstract benchmark runner: download -> ingest -> query -> evaluate."""

from __future__ import annotations

import json
import logging  # allow-direct-logging
from abc import ABC, abstractmethod
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("rag-benchmarks")


class BenchmarkRunner(ABC):
    """Base class for all benchmark adapters."""

    name: str = "base"

    def __init__(
        self,
        client: OpenAI,
        base_url: str,
        model: str,
        data_dir: str,
        output_dir: str,
        search_mode: str | None = None,
        max_queries: int | None = None,
        resume: bool = False,
        use_batch_api: bool = False,
        batch_id: str | None = None,
        extra_body: dict | None = None,
    ):
        self.client = client
        self.base_url = base_url
        self.model = model
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.search_mode = search_mode
        self.max_queries = max_queries
        self.resume = resume
        self.use_batch_api = use_batch_api
        self.batch_id = batch_id
        self.extra_body = extra_body

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> None:
        """Download the benchmark dataset."""

    @abstractmethod
    def load_data(self) -> None:
        """Load corpus, queries, and ground truth from downloaded data."""

    @abstractmethod
    def ingest(self) -> None:
        """Upload corpus to vector store via Files API + Vector Stores API."""

    @abstractmethod
    def evaluate(self) -> dict:
        """Run queries and compute metrics. Returns metrics dict."""

    def run(self) -> dict:
        """Full pipeline: download -> load -> ingest -> evaluate."""
        logger.info(f"=== Running {self.name} benchmark ===")

        logger.info("Step 1: Download")
        self.download()

        logger.info("Step 2: Load data")
        self.load_data()

        logger.info("Step 3: Ingest corpus")
        self.ingest()

        logger.info("Step 4: Evaluate")
        metrics = self.evaluate()

        # Save results
        results_path = self.output_dir / "metrics.json"
        results_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Results saved to {results_path}")

        self._print_metrics(metrics)
        return metrics

    def _load_per_query_results(self) -> dict:
        """Load existing per_query_results.json if resuming, else empty dict."""
        path = self.output_dir / "per_query_results.json"
        if self.resume and path.exists():
            data = json.loads(path.read_text())
            logger.info(f"Resumed {len(data)} existing results from {path}")
            return data
        return {}

    def _save_per_query_results(self, results: dict) -> None:
        """Write per_query_results.json with the full accumulated results."""
        path = self.output_dir / "per_query_results.json"
        path.write_text(json.dumps(results, indent=2))

    def _load_per_query_retrieval_results(self) -> dict:
        """Load existing per_query_retrieval_results.json if resuming."""
        path = self.output_dir / "per_query_retrieval_results.json"
        if self.resume and path.exists():
            data = json.loads(path.read_text())
            logger.info(f"Resumed {len(data)} existing retrieval results from {path}")
            return data
        return {}

    def _save_per_query_retrieval_results(self, results: dict) -> None:
        """Write per_query_retrieval_results.json."""
        path = self.output_dir / "per_query_retrieval_results.json"
        path.write_text(json.dumps(results, indent=2))

    def _print_metrics(self, metrics: dict) -> None:
        """Pretty-print metrics summary."""
        logger.info(f"--- {self.name} Results ---")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
