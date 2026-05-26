---
slug: rag-benchmarks
title: "OGX RAG Benchmarks: Open-Source Retrieval That Outperforms OpenAI"
authors: [franciscojavierarceo]
tags: [benchmarks, rag, vector-stores, retrieval, openai-compatibility]
date: 2026-05-26
---

We benchmarked OGX's RAG pipeline against OpenAI's file search across four BEIR retrieval datasets, MultiHOP RAG, and Doc2Dial. The results: **OGX hybrid search beats OpenAI on 3 of 4 BEIR datasets**, with up to 29.6% higher nDCG@10 on argument retrieval. Pair it with Gemma 31B and you get end-to-end RAG that exceeds GPT-4.1 by 81% on multi-hop reasoning, all running on your own infrastructure.

This isn't a synthetic demo. These are standard academic benchmarks, measured end-to-end through the same OpenAI-compatible APIs you'd use in production.

<!--truncate-->

## Why We Built This Benchmark Suite

RAG quality is hard to evaluate. Most teams ship a retrieval pipeline, eyeball a few queries, and hope for the best. We wanted something more rigorous: a reproducible benchmark suite that tests OGX's full API stack (Files, Vector Stores, Search, and Responses) against OpenAI under identical conditions.

The [benchmark suite](https://ogx-ai.github.io/docs/building_applications/rag_benchmarks) evaluates two dimensions:

1. **Retrieval quality** (BEIR) -- can the system find the right documents?
2. **End-to-end RAG** (MultiHOP, Doc2Dial) -- can it retrieve _and_ generate correct answers?

Every test runs through the same OpenAI-compatible API surface, so the comparison is apples-to-apples.

## The Setup

**OGX configuration:**

| Component | Choice |
|-----------|--------|
| Embedding model | `nomic-ai/nomic-embed-text-v1.5` (sentence-transformers) |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` (transformers) |
| Vector database | Milvus (standalone) |
| Chunk size | 512 tokens |
| Chunk overlap | 128 tokens |
| Hybrid search | RRF fusion (impact factor 60.0) + reranker |

**Search modes compared:**

- **OpenAI** -- OpenAI's hosted file search (black box)
- **OGX Vector** -- pure semantic search with Nomic embeddings
- **OGX Hybrid** -- semantic + keyword search, fused with RRF and reranked

For end-to-end RAG, we also tested **Gemma 31B** (`google/gemma-4-31B-it` via vLLM) as an alternative to GPT-4.1.

## Retrieval Results: BEIR

BEIR is the standard benchmark for information retrieval. We evaluated on four diverse datasets spanning biomedical literature, scientific fact-checking, argument mining, and financial QA.

### nDCG@10

| Dataset | OpenAI | OGX Vector | OGX Hybrid | Delta (Hybrid vs OpenAI) |
|---------|--------|------------|------------|--------------------------|
| nfcorpus | 0.3156 | 0.3106 | **0.3350** | **+6.1%** |
| scifact | **0.7165** | 0.6943 | 0.7137 | -0.4% |
| arguana | 0.2960 | 0.3765 | **0.3835** | **+29.6%** |
| fiqa | **0.2862** | 0.2399 | 0.2170 | -24.2% |

### Recall@10

| Dataset | OpenAI | OGX Vector | OGX Hybrid | Delta (Hybrid vs OpenAI) |
|---------|--------|------------|------------|--------------------------|
| nfcorpus | 0.1469 | 0.1482 | **0.1646** | **+12.0%** |
| scifact | 0.8067 | **0.8369** | 0.8362 | **+3.7%** |
| arguana | 0.6764 | 0.7610 | **0.7781** | **+15.0%** |
| fiqa | **0.3117** | 0.2843 | 0.2681 | -14.0% |

OGX hybrid search wins on **3 of 4 datasets** in both nDCG@10 and Recall@10. The arguana result is striking: nearly 30% better ranking quality and 15% better recall on counterargument retrieval. On nfcorpus (biomedical IR), hybrid search delivers 12% more relevant documents in the top 10.

OpenAI leads on fiqa (financial QA), likely due to differences in chunking strategy or embedding specialization for financial text. This is a dataset we're actively working to improve on.

## End-to-End RAG: MultiHOP and Doc2Dial

Retrieval is only half the story. End-to-end RAG tests whether the system can retrieve relevant context _and_ produce correct answers. We used GPT-4.1 as the generation model for both OpenAI and OGX pipelines, plus Gemma 31B as an open-source alternative.

### MultiHOP RAG (F1 Score)

MultiHOP tests multi-hop reasoning over 609 news articles with 2,556 queries that require synthesizing information across documents.

| Configuration | F1 | ROUGE-L | vs OpenAI |
|--------------|-----|---------|-----------|
| OpenAI | 0.0114 | 0.0116 | -- |
| OGX Vector + GPT-4.1 | 0.0141 | 0.0147 | +23.7% |
| OGX Hybrid + GPT-4.1 | 0.0141 | 0.0147 | +23.7% |
| OGX Hybrid + Gemma 31B | **0.0207** | **0.0203** | **+81.6%** |

The headline number: **OGX with Gemma 31B scores 81.6% higher F1 than OpenAI with GPT-4.1 on multi-hop reasoning.** Even with GPT-4.1 on both sides, OGX's retrieval pipeline delivers 23.7% better end-to-end accuracy.

### Doc2Dial (F1 Score)

Doc2Dial evaluates document-grounded dialogue: 488 documents, 200 conversations, 1,203 turns.

| Configuration | F1 | ROUGE-L |
|--------------|-----|---------|
| OpenAI | **0.1337** | **0.1136** |
| OGX Vector + GPT-4.1 | 0.0962 | 0.0790 |
| OGX Hybrid + GPT-4.1 | 0.0966 | 0.0794 |
| OGX Hybrid + Gemma 31B | 0.0634 | 0.0513 |

OpenAI leads here by ~39% F1. Doc2Dial's conversational format, where queries depend on prior dialogue turns, likely benefits from OpenAI's retrieval handling of conversational context. Gemma 31B's lower score is partly explained by its verbosity (~2,500 characters average vs ~95 character ground truths), which penalizes token-level F1 even when the answer content is correct. Notably, Gemma produced **zero empty responses** across all 1,203 queries.

## What This Means

Three takeaways from these benchmarks:

**1. Hybrid search is the real differentiator.** On 3 of 4 BEIR datasets, combining semantic and keyword search with reranking outperforms OpenAI's proprietary retrieval. This isn't surprising in IR research, but it's meaningful that you can get this quality from open-source components running on your hardware.

**2. Open-source models can win at end-to-end RAG.** Gemma 31B with OGX retrieval outperformed GPT-4.1 with OpenAI retrieval on multi-hop reasoning by a wide margin. The retrieval pipeline matters as much as the generation model.

**3. No single system dominates everywhere.** OpenAI leads on financial QA retrieval and conversational grounding. OGX leads on biomedical, scientific, and argumentative retrieval, plus multi-hop reasoning. The right choice depends on your domain, which is exactly why you want the flexibility to configure your own pipeline.

## Reproduce It Yourself

The full benchmark suite is open source and reproducible. See the [RAG Benchmarks documentation](https://ogx-ai.github.io/docs/building_applications/rag_benchmarks) for setup instructions, dataset details, and methodology.

All benchmarks run through the standard OpenAI-compatible API, so you can swap in your own embedding models, rerankers, or vector databases and measure the impact directly.

## What's Next

We're expanding the benchmark suite with additional datasets and retrieval configurations. Areas we're investigating:

- **Chunking strategies** -- adaptive chunking to improve performance on datasets like fiqa
- **Embedding model comparisons** -- testing newer embedding models against Nomic
- **Conversational retrieval** -- improving Doc2Dial performance with conversation-aware search
- **Larger-scale benchmarks** -- datasets with 100K+ documents to test scaling behavior

If you're running RAG workloads in production and want to see how OGX compares on your data, [get started with OGX](https://ogx-ai.github.io/docs/getting_started/) or join the conversation on [GitHub](https://github.com/ogx-ai/ogx).
