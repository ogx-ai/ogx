# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""MultiHOP RAG benchmark — end-to-end RAG evaluation with multi-hop queries."""

from __future__ import annotations

import logging  # allow-direct-logging

from datasets import load_dataset
from lib.ingest import ingest_corpus
from lib.metrics import answer_metrics, retrieval_metrics
from lib.query import rag_query_batch
from lib.utils import IDMapping

from benchmarks.base import BenchmarkRunner

logger = logging.getLogger("rag-benchmarks")


class MultiHOPBenchmark(BenchmarkRunner):
    """MultiHOP RAG benchmark — multi-hop queries over news articles."""

    name = "multihop"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.corpus: dict[str, dict] = {}
        self.queries: dict[str, str] = {}
        self.ground_truths: dict[str, str] = {}
        self.evidence_docs: dict[str, list[str]] = {}
        self.vector_store_id: str | None = None
        self.mapping: IDMapping | None = None

    def download(self) -> None:
        cache_dir = str(self.data_dir / "multihop")
        self._qa_dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", cache_dir=cache_dir)
        self._corpus_dataset = load_dataset("yixuantt/MultiHopRAG", "corpus", cache_dir=cache_dir)
        logger.info("Downloaded MultiHopRAG dataset")

    def load_data(self) -> None:
        # Extract corpus from the dedicated corpus config
        corpus_split = list(self._corpus_dataset.keys())[0]
        for idx, item in enumerate(self._corpus_dataset[corpus_split]):
            doc_id = f"doc_{idx}"
            self.corpus[doc_id] = {
                "title": item.get("title", ""),
                "text": item.get("body", ""),
            }

        # Extract queries and ground truths from QA config
        qa_split = "test" if "test" in self._qa_dataset else "train"
        for idx, item in enumerate(self._qa_dataset[qa_split]):
            qid = str(idx)
            self.queries[qid] = item["query"]
            self.ground_truths[qid] = item["answer"]

            # Map evidence docs for retrieval evaluation
            evidence_titles = []
            for ev in item.get("evidence_list", []):
                evidence_titles.append(ev.get("title", ""))
            self.evidence_docs[qid] = evidence_titles

        logger.info(f"Loaded MultiHopRAG: {len(self.corpus)} docs, {len(self.queries)} queries")

    def ingest(self) -> None:
        checkpoint_path = str(self.output_dir / "checkpoint.json")
        self.vector_store_id, self.mapping = ingest_corpus(
            client=self.client,
            corpus=self.corpus,
            vector_store_name="multihop-rag",
            checkpoint_path=checkpoint_path,
            resume=self.resume,
        )

    def evaluate(self) -> dict:
        queries = self.queries
        if self.max_queries:
            query_ids = list(queries.keys())[: self.max_queries]
            queries = {qid: queries[qid] for qid in query_ids}

        logger.info(f"Evaluating {len(queries)} multi-hop queries...")

        per_query = self._load_per_query_results()

        results = rag_query_batch(
            client=self.client,
            model=self.model,
            queries=queries,
            vector_store_ids=[self.vector_store_id],
            mapping=self.mapping,
            max_num_results=20,
            search_mode=self.search_mode,
            use_batch_api=self.use_batch_api,
            batch_id=self.batch_id,
            extra_body=self.extra_body,
            checkpoint_path=str(self.output_dir / "query_checkpoint.json"),
        )

        for qid, r in results.items():
            per_query[qid] = {
                "prediction": r["answer"],
                "ground_truth": self.ground_truths[qid],
            }

        self._save_per_query_results(per_query)

        all_predictions = {qid: r["prediction"] for qid, r in per_query.items()}
        all_ground_truths = {qid: r["ground_truth"] for qid, r in per_query.items()}
        metrics = answer_metrics(all_predictions, all_ground_truths)

        # Retrieval metrics against evidence docs
        if self.evidence_docs:
            title_to_id = {}
            for doc_id, doc in self.corpus.items():
                title_to_id[doc.get("title", "")] = doc_id

            qrels = {}
            for qid in queries:
                qrels[qid] = {}
                for title in self.evidence_docs.get(qid, []):
                    did = title_to_id.get(title)
                    if did:
                        qrels[qid][did] = 1

            retrieved = {qid: results[qid]["retrieved_docs"] for qid in queries if qid in results}
            ret_metrics = retrieval_metrics(qrels, retrieved, k_values=[5, 10, 20])
            metrics.update(ret_metrics)

        metrics["dataset"] = "multihop"
        metrics["num_corpus_docs"] = len(self.corpus)
        metrics["search_mode"] = self.search_mode or "default"

        return metrics
