# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import time
from io import BytesIO

from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)

from ogx.log import get_logger
from ogx_api.inference.models import (
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIImageURL,
    OpenAIUserMessageParam,
)

log = get_logger(name=__name__, category="providers::file_processors")


class OgxInferenceVlmEngine(BaseVlmEngine):
    """VLM engine that routes inference through the stack's model-serving API.

    Instead of making HTTP requests to an external endpoint, this engine calls
    the stack's Inference API directly in-process via dependency injection.
    """

    def __init__(self, inference_api, model: str, event_loop: asyncio.AbstractEventLoop) -> None:
        self.inference_api = inference_api
        self.model = model
        self.event_loop = event_loop
        self._initialized = True

    def initialize(self) -> None:
        pass

    def predict_batch(self, input_batch: list[VlmEngineInput]) -> list[VlmEngineOutput]:
        if not input_batch:
            return []

        log.info(
            "Processing VLM batch through stack inference API",
            model=self.model,
            batch_size=len(input_batch),
        )

        start_time = time.time()
        outputs = [self._predict_single(inp) for inp in input_batch]
        total_time = time.time() - start_time

        log.info(
            "VLM batch complete",
            batch_size=len(input_batch),
            total_time_s=round(total_time, 2),
            per_image_s=round(total_time / len(input_batch), 2),
        )

        return outputs

    def _predict_single(self, input_data: VlmEngineInput) -> VlmEngineOutput:
        request_start = time.time()

        try:
            image = input_data.image.copy().convert("RGBA")
            img_io = BytesIO()
            image.save(img_io, "PNG")
            image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
        except Exception:
            log.exception("Failed to encode image for VLM inference")
            return VlmEngineOutput(text="", stop_reason="error")

        message = OpenAIUserMessageParam(
            role="user",
            content=[
                OpenAIChatCompletionContentPartImageParam(
                    image_url=OpenAIImageURL(url=f"data:image/png;base64,{image_base64}"),
                ),
                OpenAIChatCompletionContentPartTextParam(text=input_data.prompt),
            ],
        )

        try:
            request = OpenAIChatCompletionRequestWithExtraBody(
                model=self.model,
                messages=[message],
                temperature=input_data.temperature,
                max_tokens=input_data.max_new_tokens or None,
                stop=input_data.stop_strings or None,
                stream=False,
            )
            future = asyncio.run_coroutine_threadsafe(
                self.inference_api.openai_chat_completion(request),
                self.event_loop,
            )
            response = future.result(timeout=120)

            generated_text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            stop_reason = "end_of_sequence"
            if finish_reason == "length":
                stop_reason = "length"
            elif finish_reason == "content_filter":
                log.warning("VLM response was filtered due to content safety policy")
                stop_reason = "content_filtered"

            generation_time = time.time() - request_start

            return VlmEngineOutput(
                text=generated_text.strip(),
                stop_reason=stop_reason,
                metadata={
                    "generation_time": generation_time,
                    "model": self.model,
                },
            )
        except Exception:
            log.exception("Failed to process VLM inference request")
            return VlmEngineOutput(text="", stop_reason="error")

    def cleanup(self) -> None:
        pass
