# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from fastapi import UploadFile

from ogx_api.file_processors import ProcessFileRequest, ProcessFileResponse

from .config import PyPDFFileProcessorConfig
from .pypdf import PyPDFFileProcessor


class PyPDFFileProcessorAdapter:
    """Adapter for PyPDF file processor."""

    def __init__(self, config: PyPDFFileProcessorConfig, files_api) -> None:
        self.config = config
        self.files_api = files_api
        self.processor = PyPDFFileProcessor(config, files_api)

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using PyPDF processor."""
        return await self.processor.process_file(
            file=file,
            file_id=request.file_id,
            options=request.options,
            chunking_strategy=request.chunking_strategy,
        )

    async def shutdown(self) -> None:
        """Shutdown the PyPDF file processor."""
        pass
