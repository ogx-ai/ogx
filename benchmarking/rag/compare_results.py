#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Compare results across backends: OpenAI SaaS vs OGX (vector vs hybrid)."""

import json
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Benchmark -> primary metric mapping
PRIMARY_METRICS = {
    "beir": "ndcg_cut_10",
    "multihop": "f1",
    "qrecc": "f1",
    "doc2dial": "f1",
}

SECONDARY_METRICS = {
    "beir": ["recall_10", "map_cut_10"],
    "multihop": ["exact_match", "rouge_l", "ndcg_cut_10"],
    "qrecc": ["exact_match", "rouge_l"],
    "doc2dial": ["exact_match", "rouge_l"],
}


def load_metrics(result_dir: Path) -> dict | None:
    metrics_file = result_dir / "metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    return None


def find_results() -> list[dict]:
    """Scan results directory and collect all benchmark results."""
    rows = []
    if not RESULTS_DIR.exists():
        return rows

    for backend_dir in sorted(RESULTS_DIR.iterdir()):
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name  # "openai" or "ogx"

        for bench_dir in sorted(backend_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            benchmark = bench_dir.name

            # Check for dataset subdirs (BEIR) or direct metrics
            has_subdirs = any(d.is_dir() for d in bench_dir.iterdir())
            if has_subdirs:
                for dataset_dir in sorted(bench_dir.iterdir()):
                    if not dataset_dir.is_dir():
                        continue

                    # Check for search mode subdirs
                    metrics = load_metrics(dataset_dir)
                    if metrics:
                        rows.append(
                            {
                                "backend": backend,
                                "benchmark": benchmark,
                                "dataset": dataset_dir.name,
                                "search_mode": metrics.get("search_mode", "default"),
                                "metrics": metrics,
                            }
                        )
                    else:
                        for mode_dir in sorted(dataset_dir.iterdir()):
                            if mode_dir.is_dir():
                                m = load_metrics(mode_dir)
                                if m:
                                    rows.append(
                                        {
                                            "backend": backend,
                                            "benchmark": benchmark,
                                            "dataset": mode_dir.parent.name,
                                            "search_mode": m.get("search_mode", mode_dir.name),
                                            "metrics": m,
                                        }
                                    )
            else:
                metrics = load_metrics(bench_dir)
                if metrics:
                    rows.append(
                        {
                            "backend": backend,
                            "benchmark": benchmark,
                            "dataset": "-",
                            "search_mode": metrics.get("search_mode", "default"),
                            "metrics": metrics,
                        }
                    )

    return rows


@click.command()
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "json"]), default="table")
@click.option("--output", default=None, help="Output file path")
def main(fmt: str, output: str | None):
    rows = find_results()

    if not rows:
        click.echo("No results found in results/ directory.")
        click.echo("Run benchmarks first with: bash run_all.sh")
        return

    # Build comparison table
    table_rows = []
    for row in rows:
        benchmark = row["benchmark"]
        primary = PRIMARY_METRICS.get(benchmark, "f1")
        primary_val = row["metrics"].get(primary, "N/A")

        table_row = {
            "Benchmark": benchmark,
            "Dataset": row["dataset"],
            "Backend": row["backend"],
            "Search Mode": row["search_mode"],
            "Primary Metric": primary,
            "Score": f"{primary_val:.4f}" if isinstance(primary_val, float) else str(primary_val),
        }

        # Add secondary metrics
        for metric in SECONDARY_METRICS.get(benchmark, []):
            val = row["metrics"].get(metric, "N/A")
            table_row[metric] = f"{val:.4f}" if isinstance(val, float) else str(val)

        table_rows.append(table_row)

    df = pd.DataFrame(table_rows)

    if fmt == "csv":
        result = df.to_csv(index=False)
    elif fmt == "json":
        result = df.to_json(orient="records", indent=2)
    else:
        result = tabulate(df, headers="keys", tablefmt="github", showindex=False)

    if output:
        Path(output).write_text(result)
        click.echo(f"Results written to {output}")
    else:
        click.echo(result)

    # Print pivot comparison if we have multiple backends
    if len(df["Backend"].unique()) > 1:
        click.echo("\n--- Comparison Summary ---\n")
        pivot = df.pivot_table(
            index=["Benchmark", "Dataset"],
            columns=["Backend", "Search Mode"],
            values="Score",
            aggfunc="first",
        )
        click.echo(tabulate(pivot, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
