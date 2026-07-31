# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import click
from rich import print as rprint
from tqdm.rich import tqdm

from ..common.utils import create_bar_chart
from .utils import (
    aggregate_accuracy,
    aggregate_average,
    aggregate_categorical_count,
    aggregate_median,
    aggregate_weighted_average,
)


@click.command("run-benchmark")
@click.help_option("-h", "--help")
@click.argument("benchmark-ids", nargs=-1, required=True)
@click.option(
    "--model-id",
    required=True,
    help="model id to run the benchmark eval on",
    default=None,
    type=str,
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to the dump eval results output directory",
)
@click.option(
    "--num-examples",
    required=False,
    help="Number of examples to evaluate on, useful for debugging",
    default=None,
    type=int,
)
@click.option(
    "--temperature",
    required=False,
    help="temperature in the sampling params to run generation",
    default=0.0,
    type=float,
)
@click.option(
    "--max-tokens",
    required=False,
    help="max-tokens in the sampling params to run generation",
    default=4096,
    type=int,
)
@click.option(
    "--top-p",
    required=False,
    help="top-p in the sampling params to run generation",
    default=0.9,
    type=float,
)
@click.option(
    "--repeat-penalty",
    required=False,
    help="repeat-penalty in the sampling params to run generation",
    default=1.0,
    type=float,
)
@click.option(
    "--visualize",
    is_flag=True,
    default=False,
    help="Visualize evaluation results after completion",
)
@click.pass_context
def run_benchmark(
    ctx,
    benchmark_ids: tuple[str, ...],
    model_id: str,
    output_dir: str,
    num_examples: int | None,
    temperature: float,
    max_tokens: int,
    top_p: float,
    repeat_penalty: float,
    visualize: bool,
):
    """Run a evaluation benchmark task"""

    client = ctx.obj["client"]

    for benchmark_id in benchmark_ids:
        benchmark = client.benchmarks.retrieve(benchmark_id=benchmark_id)
        scoring_functions = benchmark.scoring_functions
        dataset_id = benchmark.dataset_id

        results = client.datasets.iterrows(dataset_id=dataset_id, limit=-1 if num_examples is None else num_examples)

        output_res = {}

        for i, r in enumerate(tqdm(results.data)):
            eval_res = client.eval.evaluate_rows(
                benchmark_id=benchmark_id,
                input_rows=[r],
                scoring_functions=scoring_functions,
                benchmark_config={
                    "type": "benchmark",
                    "eval_candidate": {
                        "type": "model",
                        "model": model_id,
                        "sampling_params": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "repeat_penalty": repeat_penalty,
                        },
                    },
                },
            )
            for k in r.keys():
                if k not in output_res:
                    output_res[k] = []
                output_res[k].append(r[k])

            for k in eval_res.generations[0].keys():
                if k not in output_res:
                    output_res[k] = []
                output_res[k].append(eval_res.generations[0][k])

            for scoring_fn in scoring_functions:
                if scoring_fn not in output_res:
                    output_res[scoring_fn] = []
                output_res[scoring_fn].append(eval_res.scores[scoring_fn].score_rows[0])

                aggregation_functions = client.scoring_functions.retrieve(
                    scoring_fn_id=scoring_fn
                ).params.aggregation_functions

                # only output the aggregation result for the last row
                if i == len(results.data) - 1:
                    for aggregation_function in aggregation_functions:
                        scoring_results = output_res[scoring_fn]
                        if aggregation_function == "categorical_count":
                            output_res[scoring_fn].append(aggregate_categorical_count(scoring_results))
                        elif aggregation_function == "average":
                            output_res[scoring_fn].append(aggregate_average(scoring_results))
                        elif aggregation_function == "weighted_average":
                            output_res[scoring_fn].append(aggregate_weighted_average(scoring_results))
                        elif aggregation_function == "median":
                            output_res[scoring_fn].append(aggregate_median(scoring_results))
                        elif aggregation_function == "accuracy":
                            output_res[scoring_fn].append(aggregate_accuracy(scoring_results))
                        else:
                            raise NotImplementedError(
                                f"Aggregation function {aggregation_function} is not supported yet"
                            )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Save results to JSON file
        output_file = os.path.join(output_dir, f"{benchmark_id}_results.json")
        with open(output_file, "w") as f:
            json.dump(output_res, f, indent=2)

        rprint(f"[green]✓[/green] Results saved to: [blue]{output_file}[/blue]!\n")

        if visualize:
            for scoring_fn in scoring_functions:
                aggregation_functions = client.scoring_functions.retrieve(
                    scoring_fn_id=scoring_fn
                ).params.aggregation_functions

                for aggregation_function in aggregation_functions:
                    res = output_res[scoring_fn]
                    assert len(res) > 0 and "score" in res[0]
                    if aggregation_function == "categorical_count":
                        scores = [str(r["score"]) for r in res]
                        unique_scores = sorted(set(scores))
                        counts = [scores.count(s) for s in unique_scores]
                        create_bar_chart(
                            counts,
                            unique_scores,
                            title=f"{scoring_fn}-{aggregation_function}",
                        )
                    else:
                        raise NotImplementedError(
                            f"Aggregation function {aggregation_function} ius not supported for visualization yet"
                        )
