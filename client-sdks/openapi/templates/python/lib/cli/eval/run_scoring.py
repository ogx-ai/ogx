# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os

import click
import pandas
from rich import print as rprint
from tqdm.rich import tqdm


@click.command("run-scoring")
@click.help_option("-h", "--help")
@click.argument("scoring-function-ids", nargs=-1, required=True)
@click.option(
    "--dataset-id",
    required=False,
    help="Pre-registered dataset_id to score (from ogx-client datasets list)",
)
@click.option(
    "--dataset-path",
    required=False,
    help="Path to the dataset file to score",
    type=click.Path(exists=True),
)
@click.option(
    "--scoring-params-config",
    required=False,
    help="Path to the scoring params config file in JSON format",
    type=click.Path(exists=True),
)
@click.option(
    "--num-examples",
    required=False,
    help="Number of examples to evaluate on, useful for debugging",
    default=None,
    type=int,
)
@click.option(
    "--output-dir",
    required=True,
    help="Path to the dump eval results output directory",
)
@click.option(
    "--visualize",
    is_flag=True,
    default=False,
    help="Visualize evaluation results after completion",
)
@click.pass_context
def run_scoring(
    ctx,
    scoring_function_ids: tuple[str, ...],
    dataset_id: str | None,
    dataset_path: str | None,
    scoring_params_config: str | None,
    num_examples: int | None,
    output_dir: str,
):
    """Run scoring from application datasets"""
    # one of dataset_id or dataset_path is required
    if dataset_id is None and dataset_path is None:
        raise click.BadParameter("Specify either dataset_id (pre-registered dataset) or dataset_path (local file)")

    client = ctx.obj["client"]

    scoring_params = dict.fromkeys(scoring_function_ids)
    if scoring_params_config:
        with open(scoring_params_config) as f:
            scoring_params = json.load(f)

    output_res = {}

    if dataset_id is not None:
        dataset = client.datasets.retrieve(dataset_id=dataset_id)
        if not dataset:
            click.BadParameter(f"Dataset {dataset_id} not found. Please register using ogx-client datasets register")

        # TODO: this will eventually be replaced with jobs polling from server vis score_bath
        # For now, get all datasets rows via datasets API
        results = client.datasets.iterrows(dataset_id=dataset_id, limit=-1 if num_examples is None else num_examples)
        rows = results.rows

    if dataset_path is not None:
        df = pandas.read_csv(dataset_path)
        rows = df.to_dict(orient="records")
        if num_examples is not None:
            rows = rows[:num_examples]

    for r in tqdm(rows):
        score_res = client.scoring.score(
            input_rows=[r],
            scoring_functions=scoring_params,
        )
        for k in r.keys():
            if k not in output_res:
                output_res[k] = []
            output_res[k].append(r[k])

        for fn_id in scoring_function_ids:
            if fn_id not in output_res:
                output_res[fn_id] = []
            output_res[fn_id].append(score_res.results[fn_id].score_rows[0])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_path or dataset_id}_score_results.csv")
    df = pandas.DataFrame(output_res)
    df.to_csv(output_file, index=False)
    print(df)

    rprint(f"[green]✓[/green] Results saved to: [blue]{output_file}[/blue]!\n")
