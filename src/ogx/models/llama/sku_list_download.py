# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass

from .sku_types import (
    CheckpointQuantizationFormat,
    CoreModelId,
    Model,
    ModelFamily,
)


@dataclass
class LlamaDownloadInfo:
    """Download metadata for retrieving a Llama model from llamameta.net."""

    folder: str
    files: list[str]
    pth_size: int


def llama_meta_net_info(model: Model) -> LlamaDownloadInfo:
    """Information needed to download model from llamameta.net"""

    pth_count = model.pth_file_count
    if model.core_model_id == CoreModelId.llama3_1_405b:
        if pth_count == 16:
            folder = "Llama-3.1-405B-MP16"
        elif model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            folder = "Llama-3.1-405B"
        else:
            folder = "Llama-3.1-405B-MP8"
    elif model.core_model_id == CoreModelId.llama3_1_405b_instruct:
        if pth_count == 16:
            folder = "Llama-3.1-405B-Instruct-MP16"
        elif model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            folder = "Llama-3.1-405B-Instruct"
        else:
            folder = "Llama-3.1-405B-Instruct-MP8"
    elif model.core_model_id == CoreModelId.llama_guard_3_8b:
        if model.quantization_format == CheckpointQuantizationFormat.int8:
            folder = "Llama-Guard-3-8B-INT8-HF"
        else:
            folder = "Llama-Guard-3-8B"
    elif model.core_model_id == CoreModelId.llama_guard_2_8b:
        folder = "llama-guard-2"
    else:
        if model.huggingface_repo is None:
            raise ValueError(f"Model {model.core_model_id} has no huggingface_repo set")
        folder = model.huggingface_repo.split("/")[-1]
        if "Llama-2" in folder:
            folder = folder.lower()

    files = ["checklist.chk"]
    if (
        model.core_model_id == CoreModelId.llama_guard_3_8b
        and model.quantization_format == CheckpointQuantizationFormat.int8
    ):
        files.extend(
            [
                "generation_config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            ]
        )
    elif (
        model.core_model_id == CoreModelId.llama_guard_3_1b
        and model.quantization_format == CheckpointQuantizationFormat.int4
    ):
        files.extend(
            [
                "llama_guard_3_1b_pruned_xnnpack.pte",
                "example-prompt.txt",
                "params.json",
                "tokenizer.model",
            ]
        )
    else:
        files.extend(
            [
                "tokenizer.model",
                "params.json",
            ]
        )
        if model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            files.extend([f"fp8_scales_{i}.pt" for i in range(pth_count)])
        files.extend([f"consolidated.{i:02d}.pth" for i in range(pth_count)])

    return LlamaDownloadInfo(
        folder=folder,
        files=files,
        pth_size=llama_meta_pth_size(model),
    )


# Sadness because Cloudfront rejects our HEAD requests to find Content-Length
def llama_meta_pth_size(model: Model) -> int:
    if model.core_model_id not in (
        CoreModelId.llama3_1_405b,
        CoreModelId.llama3_1_405b_instruct,
        CoreModelId.llama4_maverick_17b_128e,
        CoreModelId.llama4_maverick_17b_128e_instruct,
    ):
        return 0

    if model.model_family == ModelFamily.llama3_1:
        if model.pth_file_count == 16:
            return 51268302389
        elif model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            return 60903742309
        else:
            return 101470976045

    if model.model_family == ModelFamily.llama4:
        if model.core_model_id == CoreModelId.llama4_maverick_17b_128e:
            return 100458118386
        elif model.core_model_id == CoreModelId.llama4_maverick_17b_128e_instruct:
            if model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
                return 54121549657
            else:
                return 100426653046
    return 0
