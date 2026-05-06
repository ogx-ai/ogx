# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from ogx.log import get_logger
from ogx_api import OpenAIMessageParam, RunShieldRequest, Safety, SafetyViolation, ViolationLevel

log = get_logger(name=__name__, category="agents::builtin")


class SafetyException(Exception):  # noqa: N818
    """Raised when a safety shield detects a policy violation."""

    def __init__(self, violation: SafetyViolation):
        self.violation = violation
        super().__init__(violation.user_message)


class ShieldRunnerMixin:
    """Mixin for running input and output safety shields on messages."""

    def __init__(
        self,
        safety_api: Safety,
        input_shields: list[str] | None = None,
        output_shields: list[str] | None = None,
    ):
        self.safety_api = safety_api
        self.input_shields = input_shields
        self.output_shields = output_shields

    async def run_multiple_shields(self, messages: list[OpenAIMessageParam], identifiers: list[str]) -> None:
        responses = await asyncio.gather(
            *[
                self.safety_api.run_shield(RunShieldRequest(shield_id=identifier, messages=messages))
                for identifier in identifiers
            ]
        )
        for identifier, response in zip(identifiers, responses, strict=False):
            if not response.violation:
                continue

            violation = response.violation
            if violation.violation_level == ViolationLevel.ERROR:
                raise SafetyException(violation)
            elif violation.violation_level == ViolationLevel.WARN:
                log.warning(f"[Warn]{identifier} raised a warning")
