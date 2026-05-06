#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""CLI entry point for running RAG benchmarks."""

from pathlib import Path

import click
from lib.client import OPENAI_SAAS_URL, backend_label, create_client
from lib.utils import setup_logging


@click.command()
@click.option("--benchmark", type=click.Choice(["beir", "multihop", "qrecc", "doc2dial"]), required=True)
@click.option("--dataset", default=None, help="Dataset name (for BEIR: nfcorpus, scifact, arguana, fiqa, trec-covid)")
@click.option("--base-url", default=OPENAI_SAAS_URL, help="API base URL")
@click.option("--search-mode", type=click.Choice(["vector", "hybrid", "keyword"]), default=None)
@click.option("--model", default="gpt-4.1", help="Model for end-to-end RAG")
@click.option("--max-queries", type=int, default=None, help="Limit number of queries (for testing)")
@click.option("--max-conversations", type=int, default=None, help="Limit conversations (QReCC/Doc2Dial)")
@click.option("--resume/--no-resume", default=False, help="Resume from checkpoint")
@click.option("--output-dir", default=None, help="Output directory")
@click.option("--data-dir", default=None, help="Data directory")
@click.option("--verbose", is_flag=True, default=False)
@click.option(
    "--batch-api/--no-batch-api",
    default=False,
    help="Use OpenAI Batch API for queries (50% cheaper, higher throughput)",
)
@click.option("--batch-id", default=None, help="Resume polling an existing Batch API job by ID")
@click.option(
    "--disable-thinking",
    is_flag=True,
    default=False,
    help="Disable thinking/reasoning mode for models that support it (e.g. vLLM reasoning models)",
)
def main(
    benchmark: str,
    dataset: str | None,
    base_url: str,
    search_mode: str | None,
    model: str,
    max_queries: int | None,
    max_conversations: int | None,
    resume: bool,
    output_dir: str | None,
    data_dir: str | None,
    verbose: bool,
    batch_api: bool,
    batch_id: str | None,
    disable_thinking: bool,
):
    setup_logging(verbose)

    rag_dir = Path(__file__).resolve().parent
    data_dir = data_dir or str(rag_dir / "data")
    label = backend_label(base_url)

    # Default output dir
    if not output_dir:
        parts = [label, benchmark]
        if dataset:
            parts.append(dataset)
        if search_mode:
            parts.append(search_mode)
        output_dir = str(rag_dir / "results" / "/".join(parts))

    client = create_client(base_url=base_url)

    extra_body = None
    if disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    common_kwargs = {
        "client": client,
        "base_url": base_url,
        "model": model,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "search_mode": search_mode,
        "max_queries": max_queries,
        "resume": resume,
        "use_batch_api": batch_api,
        "batch_id": batch_id,
        "extra_body": extra_body,
    }

    if benchmark == "beir":
        from benchmarks.beir_bench import BEIRBenchmark

        if not dataset:
            dataset = "nfcorpus"
        runner = BEIRBenchmark(dataset=dataset, **common_kwargs)

    elif benchmark == "multihop":
        from benchmarks.multihop_bench import MultiHOPBenchmark

        runner = MultiHOPBenchmark(**common_kwargs)

    elif benchmark == "qrecc":
        from benchmarks.qrecc_bench import QReCCBenchmark

        runner = QReCCBenchmark(
            max_conversations=max_conversations,
            **common_kwargs,
        )

    elif benchmark == "doc2dial":
        from benchmarks.doc2dial_bench import Doc2DialBenchmark

        runner = Doc2DialBenchmark(
            max_conversations=max_conversations,
            **common_kwargs,
        )

    metrics = runner.run()

    # Print summary
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Benchmark: {benchmark}" + (f" / {dataset}" if dataset else ""))
    click.echo(f"Backend:   {label} ({base_url})")
    click.echo(f"Search:    {search_mode or 'default'}")
    click.echo(f"Model:     {model}")
    click.echo(f"{'=' * 60}")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            click.echo(f"  {key:25s} {value:.4f}")
        else:
            click.echo(f"  {key:25s} {value}")
    click.echo(f"{'=' * 60}")
    click.echo(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
