#!/usr/bin/env python
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Run this script:
# torchrun --nproc_per_node=8 scripts/generate_prompt_format.py meta-llama/Llama-4-17B-Omni-Instruct-BF16-16E ~/.llama/checkpoints/Llama-4-17B-Omni-Instruct-BF16-16E/ ogx.models.llama.llama4.prompts ogx/models/llama/llama4/prompt_format.md


import importlib
import os
from pathlib import Path

import fire

from ogx.models.llama.llama3.generation import Llama3
from ogx.models.llama.llama4.generation import Llama4
from ogx.models.llama.sku_list import resolve_model
from ogx_api import ModelNotFoundError

THIS_DIR = Path(__file__).parent.resolve()


def run_main(
    model_id: str,
    checkpoint_dir: str,
    module_name: str,
    output_path: str,
    llama4: bool = True,
):
    module = importlib.import_module(module_name)
    assert hasattr(module, "usecases"), f"Module {module_name} missing usecases function"

    llama_model = resolve_model(model_id)
    if not llama_model:
        raise ModelNotFoundError(model_id)

    cls = Llama4 if llama4 else Llama3
    generator = cls.build(
        ckpt_dir=checkpoint_dir,
        max_seq_len=4096,
        max_batch_size=1,
    )

    use_cases = module.usecases()
    text = ""
    for u in use_cases:
        if isinstance(u, str):
            use_case_text = f"\n{u}\n"
        else:
            use_case_text = u.to_text(generator)

        text += use_case_text
        print(use_case_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
