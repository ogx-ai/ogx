# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckpointQuantizationFormat(Enum):
    """Supported quantization formats for model checkpoints."""

    # default format
    bf16 = "bf16"

    # used for enabling fp8_rowwise inference, some weights are bf16
    fp8_mixed = "fp8-mixed"

    int8 = "int8"

    int4 = "int4"


class ModelFamily(Enum):
    """Enumeration of Llama model families by generation."""

    llama2 = "llama2"
    llama3 = "llama3"
    llama3_1 = "llama3_1"
    llama3_2 = "llama3_2"
    llama3_3 = "llama3_3"
    llama4 = "llama4"
    safety = "safety"


class CoreModelId(Enum):
    """Each of these models is a unique "SKU". These root models can be served in various garbs (especially by quantizing them)"""

    # Llama 2 family
    llama2_7b = "Llama-2-7b"
    llama2_13b = "Llama-2-13b"
    llama2_70b = "Llama-2-70b"
    llama2_7b_chat = "Llama-2-7b-chat"
    llama2_13b_chat = "Llama-2-13b-chat"
    llama2_70b_chat = "Llama-2-70b-chat"

    # Llama 3 family
    llama3_8b = "Llama-3-8B"
    llama3_70b = "Llama-3-70B"
    llama3_8b_instruct = "Llama-3-8B-Instruct"
    llama3_70b_instruct = "Llama-3-70B-Instruct"

    # Llama 3.1 family
    llama3_1_8b = "Llama3.1-8B"
    llama3_1_70b = "Llama3.1-70B"
    llama3_1_405b = "Llama3.1-405B"
    llama3_1_8b_instruct = "Llama3.1-8B-Instruct"
    llama3_1_70b_instruct = "Llama3.1-70B-Instruct"
    llama3_1_405b_instruct = "Llama3.1-405B-Instruct"

    # Llama 3.2 family
    llama3_2_1b = "Llama3.2-1B"
    llama3_2_3b = "Llama3.2-3B"
    llama3_2_1b_instruct = "Llama3.2-1B-Instruct"
    llama3_2_3b_instruct = "Llama3.2-3B-Instruct"
    llama3_2_11b_vision = "Llama3.2-11B-Vision"
    llama3_2_90b_vision = "Llama3.2-90B-Vision"
    llama3_2_11b_vision_instruct = "Llama3.2-11B-Vision-Instruct"
    llama3_2_90b_vision_instruct = "Llama3.2-90B-Vision-Instruct"

    # Llama 3.3 family
    llama3_3_70b_instruct = "Llama3.3-70B-Instruct"

    # Llama 4 family
    llama4_scout_17b_16e = "Llama-4-Scout-17B-16E"
    llama4_scout_17b_16e_instruct = "Llama-4-Scout-17B-16E-Instruct"
    llama4_maverick_17b_128e = "Llama-4-Maverick-17B-128E"
    llama4_maverick_17b_128e_instruct = "Llama-4-Maverick-17B-128E-Instruct"

    # Safety models
    llama_guard_3_8b = "Llama-Guard-3-8B"
    llama_guard_2_8b = "Llama-Guard-2-8B"
    llama_guard_3_11b_vision = "Llama-Guard-3-11B-Vision"
    llama_guard_3_1b = "Llama-Guard-3-1B"
    llama_guard_4_12b = "Llama-Guard-4-12B"


def is_multimodal(model_id) -> bool:
    """Check whether the given model supports multimodal (vision) input.

    Args:
        model_id: the CoreModelId to check.

    Returns:
        True if the model supports vision input, False otherwise.
    """
    if model_id in [
        CoreModelId.llama3_2_11b_vision,
        CoreModelId.llama3_2_90b_vision,
        CoreModelId.llama3_2_11b_vision_instruct,
        CoreModelId.llama3_2_90b_vision_instruct,
    ]:
        return True
    else:
        return False


def model_family(model_id) -> ModelFamily:
    """Determine the model family for a given CoreModelId.

    Args:
        model_id: the CoreModelId to classify.

    Returns:
        The ModelFamily that the model belongs to.
    """
    if model_id in [
        CoreModelId.llama2_7b,
        CoreModelId.llama2_13b,
        CoreModelId.llama2_70b,
        CoreModelId.llama2_7b_chat,
        CoreModelId.llama2_13b_chat,
        CoreModelId.llama2_70b_chat,
    ]:
        return ModelFamily.llama2
    elif model_id in [
        CoreModelId.llama3_8b,
        CoreModelId.llama3_70b,
        CoreModelId.llama3_8b_instruct,
        CoreModelId.llama3_70b_instruct,
    ]:
        return ModelFamily.llama3
    elif model_id in [
        CoreModelId.llama3_1_8b,
        CoreModelId.llama3_1_70b,
        CoreModelId.llama3_1_405b,
        CoreModelId.llama3_1_8b_instruct,
        CoreModelId.llama3_1_70b_instruct,
        CoreModelId.llama3_1_405b_instruct,
    ]:
        return ModelFamily.llama3_1
    elif model_id in [
        CoreModelId.llama3_2_1b,
        CoreModelId.llama3_2_3b,
        CoreModelId.llama3_2_1b_instruct,
        CoreModelId.llama3_2_3b_instruct,
        CoreModelId.llama3_2_11b_vision,
        CoreModelId.llama3_2_90b_vision,
        CoreModelId.llama3_2_11b_vision_instruct,
        CoreModelId.llama3_2_90b_vision_instruct,
    ]:
        return ModelFamily.llama3_2
    elif model_id in [
        CoreModelId.llama3_3_70b_instruct,
    ]:
        return ModelFamily.llama3_3
    elif model_id in [
        CoreModelId.llama4_scout_17b_16e,
        CoreModelId.llama4_scout_17b_16e_instruct,
        CoreModelId.llama4_maverick_17b_128e,
        CoreModelId.llama4_maverick_17b_128e_instruct,
    ]:
        return ModelFamily.llama4
    elif model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.llama_guard_2_8b,
        CoreModelId.llama_guard_3_11b_vision,
        CoreModelId.llama_guard_3_1b,
        CoreModelId.llama_guard_4_12b,
    ]:
        return ModelFamily.safety
    else:
        raise ValueError(f"Unknown model family for {model_id}")


class Model(BaseModel):
    """Represents a Llama model SKU with its architecture parameters and metadata."""

    core_model_id: CoreModelId
    description: str
    huggingface_repo: str | None = None
    arch_args: dict[str, Any]
    variant: str = ""

    quantization_format: CheckpointQuantizationFormat = CheckpointQuantizationFormat.bf16
    pth_file_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    # silence pydantic until we remove the `model_` fields
    model_config = ConfigDict(protected_namespaces=())

    @property
    def model_family(self) -> ModelFamily:
        return model_family(self.core_model_id)

    # The SKU is uniquely identified by (model_id, variant) combo
    def descriptor(self, shorten_default_variant: bool = True) -> str:
        if not self.variant:
            return self.core_model_id.value
        return f"{self.core_model_id.value}:{self.variant}"

    @property
    def is_instruct_model(self) -> bool:
        return "instruct" in self.core_model_id.value

    # Featured models are shown in the non-exhaustive model list
    @property
    def is_featured(self) -> bool:
        return self.model_family in [
            ModelFamily.llama3_1,
            ModelFamily.llama3_2,
            ModelFamily.llama3_3,
            ModelFamily.llama4,
            ModelFamily.safety,
        ]

    @property
    def max_seq_length(self) -> int:
        if self.model_family == ModelFamily.llama2:
            return 4096
        elif self.core_model_id == CoreModelId.llama_guard_2_8b:
            return 4096
        elif self.model_family == ModelFamily.llama3:
            return 8192
        elif self.model_family in [ModelFamily.llama3_1, ModelFamily.llama3_3]:
            return 131072
        elif self.model_family == ModelFamily.llama3_2:
            if self.quantization_format == CheckpointQuantizationFormat.int4:
                return 8192
            return 131072
        elif self.model_family == ModelFamily.llama4:
            if self.core_model_id in {
                CoreModelId.llama4_scout_17b_16e,
                CoreModelId.llama4_maverick_17b_128e,
            }:
                return 262144
            if self.core_model_id == CoreModelId.llama4_scout_17b_16e_instruct:
                return 10485760
            if self.core_model_id == CoreModelId.llama4_maverick_17b_128e_instruct:
                return 1048576

            raise AssertionError(f"Unexpected core model id: {self.core_model_id}")
        elif self.core_model_id in [
            CoreModelId.llama_guard_3_8b,
            CoreModelId.llama_guard_3_11b_vision,
            CoreModelId.llama_guard_3_1b,
        ]:
            return 131072
        elif self.core_model_id == CoreModelId.llama_guard_4_12b:
            return 8192
        else:
            raise ValueError(f"Unknown max_seq_len for {self.core_model_id}")
