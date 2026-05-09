# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.utils.model_utils import model_local_dir
from ogx.log import get_logger
from ogx.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from ogx.providers.utils.safety import ShieldToModerationMixin
from ogx_api import (
    GetShieldRequest,
    OpenAIMessageParam,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    Shield,
    ShieldsProtocolPrivate,
    ShieldStore,
    ViolationLevel,
)

from .config import PromptGuardConfig, PromptGuardType

log = get_logger(name=__name__, category="safety")

PROMPT_GUARD_MODEL = "Prompt-Guard-86M"


class PromptGuardSafetyImpl(ShieldToModerationMixin, Safety, ShieldsProtocolPrivate):
    """Safety provider using the Prompt Guard model to detect prompt injection and jailbreak attempts."""

    shield_store: ShieldStore

    def __init__(self, config: PromptGuardConfig, _deps) -> None:
        self.config = config

    async def initialize(self) -> None:
        # Lazy import torch and transformers to reduce startup memory (~46MB+ savings)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = model_local_dir(PROMPT_GUARD_MODEL)
        self.shield = PromptGuardShield(
            model_dir, self.config, torch, AutoModelForSequenceClassification, AutoTokenizer
        )

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if shield.provider_resource_id != PROMPT_GUARD_MODEL:
            raise ValueError(f"Only {PROMPT_GUARD_MODEL} is supported for Prompt Guard. ")

    async def unregister_shield(self, identifier: str) -> None:
        pass

    async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(GetShieldRequest(identifier=request.shield_id))
        if not shield:
            raise ValueError(f"Unknown shield {request.shield_id}")

        return await self.shield.run(request.messages)


class PromptGuardShield:
    """Runs the Prompt Guard classifier model to score messages for injection or jailbreak risk."""

    def __init__(
        self,
        model_dir: str,
        config: PromptGuardConfig,
        torch_module: Any,
        auto_model_class: Any,
        auto_tokenizer_class: Any,
        threshold: float = 0.9,
        temperature: float = 1.0,
    ):
        assert model_dir is not None, "Must provide a model directory for prompt injection shield"
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        self.config = config
        self.temperature = temperature
        self.threshold = threshold
        self.torch = torch_module

        self.device = "cpu"
        if self.torch.cuda.is_available():
            self.device = "cuda"

        # load model and tokenizer
        self.tokenizer = auto_tokenizer_class.from_pretrained(model_dir)
        self.model = auto_model_class.from_pretrained(model_dir, device_map=self.device)

    async def run(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        message = messages[-1]
        text = interleaved_content_as_str(message.content)

        # run model on messages and return response
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs[0]
        probabilities = self.torch.softmax(logits / self.temperature, dim=-1)
        score_embedded = probabilities[0, 1].item()
        score_malicious = probabilities[0, 2].item()
        log.info(
            f"Ran PromptGuardShield and got Scores: Embedded: {score_embedded}, Malicious: {score_malicious}",
        )

        violation = None
        if self.config.guard_type == PromptGuardType.injection.value and (
            score_embedded + score_malicious > self.threshold
        ):
            violation = SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message="Sorry, I cannot do this.",
                metadata={
                    "violation_type": f"prompt_injection:embedded={score_embedded},malicious={score_malicious}",
                },
            )
        elif self.config.guard_type == PromptGuardType.jailbreak.value and score_malicious > self.threshold:
            violation = SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message="Sorry, I cannot do this.",
                metadata={
                    "violation_type": f"prompt_injection:malicious={score_malicious}",
                },
            )

        return RunShieldResponse(violation=violation)
