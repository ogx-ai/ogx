# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeshield.cs import CodeShieldScanResult

from ogx.log import get_logger
from ogx.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from ogx_api import (
    GetShieldRequest,
    ModerationObject,
    ModerationObjectResults,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
    Safety,
    SafetyViolation,
    Shield,
    ViolationLevel,
)

from .config import CodeScannerConfig

log = get_logger(name=__name__, category="safety")

ALLOWED_CODE_SCANNER_MODEL_IDS = [
    "code-scanner",
    "code-shield",
]


class BuiltinCodeScannerSafetyImpl(Safety):
    """Safety provider that scans generated code for security vulnerabilities using CodeShield."""

    def __init__(self, config: CodeScannerConfig, deps) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if shield.provider_resource_id not in ALLOWED_CODE_SCANNER_MODEL_IDS:
            raise ValueError(
                f"Unsupported Code Scanner ID: {shield.provider_resource_id}. Allowed IDs: {ALLOWED_CODE_SCANNER_MODEL_IDS}"
            )

    async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(GetShieldRequest(identifier=request.shield_id))
        if not shield:
            raise ValueError(f"Shield {request.shield_id} not found")

        from codeshield.cs import CodeShield

        text = "\n".join([interleaved_content_as_str(m.content) for m in request.messages])
        log.info(f"Running CodeScannerShield on {text[50:]}")
        result = await CodeShield.scan_code(text)

        violation = None
        if result.is_insecure:
            violation = SafetyViolation(
                violation_level=(ViolationLevel.ERROR),
                user_message="Sorry, I found security concerns in the code.",
                metadata={"violation_type": ",".join([issue.pattern_id for issue in result.issues_found])},
            )
        return RunShieldResponse(violation=violation)

    def get_moderation_object_results(self, scan_result: "CodeShieldScanResult") -> ModerationObjectResults:
        categories = {}
        category_scores = {}
        category_applied_input_types = {}

        flagged = scan_result.is_insecure
        user_message = None
        metadata = {}

        if scan_result.is_insecure:
            pattern_ids = [issue.pattern_id for issue in scan_result.issues_found]
            categories = dict.fromkeys(pattern_ids, True)
            category_scores = dict.fromkeys(pattern_ids, 1.0)
            category_applied_input_types = {key: ["text"] for key in pattern_ids}
            user_message = f"Security concerns detected in the code. {scan_result.recommended_treatment.name}: {', '.join([issue.description for issue in scan_result.issues_found])}"
            metadata = {"violation_type": ",".join([issue.pattern_id for issue in scan_result.issues_found])}

        return ModerationObjectResults(
            flagged=flagged,
            categories=categories,
            category_scores=category_scores,
            category_applied_input_types=category_applied_input_types,
            user_message=user_message,
            metadata=metadata,
        )

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        if request.model is None:
            raise ValueError("Code scanner moderation requires a model identifier.")

        inputs = request.input if isinstance(request.input, list) else [request.input]
        results = []

        from codeshield.cs import CodeShield

        for text_input in inputs:
            log.info(f"Running CodeScannerShield moderation on input: {text_input[:100]}...")
            try:
                scan_result = await CodeShield.scan_code(text_input)
                moderation_result = self.get_moderation_object_results(scan_result)
            except Exception as e:
                log.error(f"CodeShield.scan_code failed: {e}")
                # create safe fallback response on scanner failure to avoid blocking legitimate requests
                moderation_result = ModerationObjectResults(
                    flagged=False,
                    categories={},
                    category_scores={},
                    category_applied_input_types={},
                    user_message=None,
                    metadata={"scanner_error": str(e)},
                )
            results.append(moderation_result)

        return ModerationObject(id=str(uuid.uuid4()), model=request.model, results=results)
