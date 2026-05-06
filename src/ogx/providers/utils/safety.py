# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import TYPE_CHECKING

from ogx_api import (
    ModerationObject,
    ModerationObjectResults,
    OpenAIUserMessageParam,
    RunModerationRequest,
    RunShieldRequest,
    RunShieldResponse,
)

if TYPE_CHECKING:
    # Type stub for mypy - actual implementation provided by provider class
    class _RunShieldProtocol:
        async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse: ...


class ShieldToModerationMixin:
    """
    Mixin that provides run_moderation implementation by delegating to run_shield.

    Providers must implement run_shield(request: RunShieldRequest) for this mixin to work.
    Providers with custom run_moderation implementations will override this automatically.
    """

    if TYPE_CHECKING:
        # Type hint for mypy - run_shield is provided by the mixed-in class
        async def run_shield(self, request: RunShieldRequest) -> RunShieldResponse: ...

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        """
        Run moderation by converting input to messages and delegating to run_shield.

        Args:
            request: RunModerationRequest with input and model

        Returns:
            ModerationObject with results for each input

        Raises:
            ValueError: If model is None
        """
        if request.model is None:
            raise ValueError(f"{self.__class__.__name__} moderation requires a model identifier")

        inputs = request.input if isinstance(request.input, list) else [request.input]
        results = []

        for text_input in inputs:
            # Convert string to OpenAI message format
            message = OpenAIUserMessageParam(content=text_input)

            # Call run_shield (must be implemented by the provider)
            shield_request = RunShieldRequest(
                shield_id=request.model,
                messages=[message],
            )
            shield_response = await self.run_shield(shield_request)

            # Convert RunShieldResponse to ModerationObjectResults
            results.append(self._shield_response_to_moderation_result(shield_response))

        return ModerationObject(
            id=f"modr-{uuid.uuid4()}",
            model=request.model,
            results=results,
        )

    def _shield_response_to_moderation_result(
        self,
        shield_response: RunShieldResponse,
    ) -> ModerationObjectResults:
        """Convert RunShieldResponse to ModerationObjectResults.

        Args:
            shield_response: The response from run_shield

        Returns:
            ModerationObjectResults with appropriate fields set
        """
        if shield_response.violation is None:
            # Safe content
            return ModerationObjectResults(
                flagged=False,
                categories={},
                category_scores={},
                category_applied_input_types={},
                user_message=None,
                metadata={},
            )

        # Unsafe content - extract violation details
        v = shield_response.violation
        violation_type = v.metadata.get("violation_type", "unsafe")

        # Ensure violation_type is a string (metadata values can be Any)
        if not isinstance(violation_type, str):
            violation_type = "unsafe"

        return ModerationObjectResults(
            flagged=True,
            categories={violation_type: True},
            category_scores={violation_type: 1.0},
            category_applied_input_types={violation_type: ["text"]},
            user_message=v.user_message,
            metadata=v.metadata,
        )
