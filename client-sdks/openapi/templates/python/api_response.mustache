"""API response object."""

from __future__ import annotations
import json as _json
from typing import Any, Generic, Mapping, TypeVar
from pydantic import Field, StrictInt, StrictBytes, BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """
    API response object.

    Supports two construction styles:
    - Normal: ApiResponse(status_code=200, headers=..., data=..., raw_data=b"...")
    - Stainless-compatible: ApiResponse(raw=httpx_response, cast_to=MyModel, ...)
      Used by OGXAsLibraryClient to wrap in-process responses.
    """

    status_code: StrictInt = Field(description="HTTP status code")
    headers: Mapping[str, str] | None = Field(None, description="HTTP headers")
    data: T = Field(default=None, description="Deserialized data given the data type")
    raw_data: StrictBytes = Field(default=b"", description="Raw data (HTTP response body)")

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, *, raw: Any = None, cast_to: Any = None, **kwargs: Any) -> None:
        # Accept and ignore stainless-specific kwargs (client, options, stream, stream_cls, retries_taken)
        for key in ("client", "options", "stream", "stream_cls", "retries_taken"):
            kwargs.pop(key, None)
        if raw is not None:
            super().__init__(
                status_code=raw.status_code,
                headers=dict(raw.headers) if raw.headers else None,
                data=None,
                raw_data=raw.content if isinstance(raw.content, bytes) else b"",
                **kwargs,
            )
            object.__setattr__(self, "_raw", raw)
            object.__setattr__(self, "_cast_to", cast_to)
        else:
            super().__init__(**kwargs)
            object.__setattr__(self, "_raw", None)
            object.__setattr__(self, "_cast_to", None)

    def parse(self, *, to: Any = None) -> Any:
        """Parse the raw response into the target type."""
        cast_to = to or getattr(self, "_cast_to", None)
        raw = getattr(self, "_raw", None)
        if raw is None or cast_to is None:
            return self.data
        data = raw.json()
        if hasattr(cast_to, "from_dict"):
            return cast_to.from_dict(data)
        if hasattr(cast_to, "model_validate"):
            return cast_to.model_validate(data)
        return data
