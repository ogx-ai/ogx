# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""BEIR benchmark adapter — retrieval-only evaluation via Vector Stores Search API."""

from __future__ import annotations

import logging  # allow-direct-logging

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from lib.ingest import ingest_corpus
from lib.metrics import retrieval_metrics
from lib.search import search_queries
from lib.utils import IDMapping

from benchmarks.base import BenchmarkRunner

logger = logging.getLogger("rag-benchmarks")

BEIR_DATASETS = {
    "nfcorpus": "nfcorpus",
    "scifact": "scifact",
    "arguana": "arguana",
    "fiqa": "fiqa",
    "trec-covid": "trec-covid",
}


class BEIRBenchmark(BenchmarkRunner):
    """BEIR retrieval benchmark — nDCG@10 evaluation."""

    name = "beir"

    def __init__(self, dataset: str = "nfcorpus", **kwargs):
        super().__init__(**kwargs)
        if dataset not in BEIR_DATASETS:
            raise ValueError(f"Unknown BEIR dataset: {dataset}. Choose from: {list(BEIR_DATASETS.keys())}")
        self.dataset = dataset
        self.corpus: dict = {}
        self.queries: dict[str, str] = {}
        self.qrels: dict[str, dict[str, int]] = {}
        self.vector_store_id: str | None = None
        self.mapping: IDMapping | None = None

    def download(self) -> None:
        dataset_dir = self.data_dir / "beir" / self.dataset
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            logger.info(f"BEIR/{self.dataset} already downloaded at {dataset_dir}")
            return

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip"
        download_dir = str(self.data_dir / "beir")
        beir_util.download_and_unzip(url, download_dir)
        logger.info(f"Downloaded BEIR/{self.dataset} to {dataset_dir}")

    def load_data(self) -> None:
        dataset_dir = str(self.data_dir / "beir" / self.dataset)
        self.corpus, self.queries, self.qrels = GenericDataLoader(dataset_dir).load(split="test")
        logger.info(
            f"Loaded BEIR/{self.dataset}: "
            f"{len(self.corpus)} docs, {len(self.queries)} queries, "
            f"{len(self.qrels)} judged queries"
        )

    def ingest(self) -> None:
        # Convert BEIR corpus format to our format
        corpus_docs = {}
        for doc_id, doc in self.corpus.items():
            corpus_docs[doc_id] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

        checkpoint_path = str(self.output_dir / "checkpoint.json")
        self.vector_store_id, self.mapping = ingest_corpus(
            client=self.client,
            corpus=corpus_docs,
            vector_store_name=f"beir-{self.dataset}",
            checkpoint_path=checkpoint_path,
            resume=self.resume,
        )

    def evaluate(self) -> dict:
        queries = self.queries
        if self.max_queries:
            query_ids = list(queries.keys())[: self.max_queries]
            queries = {qid: queries[qid] for qid in query_ids}

        logger.info(f"Evaluating {len(queries)} queries...")

        results = search_queries(
            client=self.client,
            vector_store_id=self.vector_store_id,
            queries=queries,
            mapping=self.mapping,
            max_num_results=10,
            search_mode=self.search_mode,
        )

        # Compute metrics
        metrics = retrieval_metrics(self.qrels, results, k_values=[5, 10])

        metrics["dataset"] = self.dataset
        metrics["num_corpus_docs"] = len(self.corpus)
        metrics["num_queries"] = len(queries)
        metrics["search_mode"] = self.search_mode or "default"

        return metrics
