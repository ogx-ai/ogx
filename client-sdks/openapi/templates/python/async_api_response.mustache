"""Async API response object."""

from __future__ import annotations
from typing import Generic, Mapping, TypeVar, AsyncIterator, Any
import httpx

T = TypeVar("T")


class AsyncApiResponse(Generic[T]):
    """
    Async API response object.

    Supports two construction styles:
    - Normal: AsyncApiResponse(response=httpx_response)
    - Stainless-compatible: AsyncApiResponse(raw=httpx_response, cast_to=MyModel, stream=True, stream_cls=AsyncStream[T], ...)
      Used by OGXAsLibraryClient to wrap in-process responses.
    """

    def __init__(
        self,
        response: httpx.Response | None = None,
        *,
        raw: httpx.Response | None = None,
        cast_to: Any = None,
        stream: bool = False,
        stream_cls: Any = None,
        **kwargs: Any,
    ) -> None:
        self._response = response or raw
        if self._response is None:
            raise ValueError("Either 'response' or 'raw' must be provided")
        self._data: T | None = None
        self._raw_data: bytes | None = None
        self._cast_to = cast_to
        self._stream = stream
        self._stream_cls = stream_cls

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        return self._response.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        """HTTP headers."""
        return self._response.headers

    @property
    def raw_data(self) -> bytes:
        """Raw data (HTTP response body)."""
        if self._raw_data is None:
            self._raw_data = self._response.content
        return self._raw_data

    async def read(self) -> bytes:
        """
        Asynchronously read and return binary response content.

        :return: Response content as bytes
        """
        if self._raw_data is None:
            self._raw_data = await self._response.aread()
        return self._raw_data

    async def text(self) -> str:
        """
        Asynchronously read and decode response content to a string.

        :return: Response content as string
        """
        content = await self.read()
        return content.decode(self._response.encoding or 'utf-8')

    async def json(self) -> Any:
        """
        Asynchronously read and decode JSON response content.

        :return: Decoded JSON object
        """
        return self._response.json()

    async def parse(self, cast_to: type[T] = None) -> T:
        """
        Parse the response data.

        Handles both normal and stainless-compatible construction styles.

        :param cast_to: Optional type to cast the response to
        :return: Parsed response data
        """
        # Stainless-compatible streaming: construct the stream class directly
        if self._stream and self._stream_cls:
            from typing import get_args
            args = get_args(self._stream_cls)
            chunk_type = args[0] if args else None
            return self._stream_cls(  # type: ignore
                response=self._response,
                client=None,
                cast_to=chunk_type,
            )

        # Stainless-compatible non-streaming: use stored cast_to
        effective_cast_to = cast_to or self._cast_to
        if effective_cast_to is None:
            return await self.json()  # type: ignore

        # Handle different response types
        if effective_cast_to == bytes:
            return await self.read()  # type: ignore
        elif effective_cast_to == str:
            return await self.text()  # type: ignore
        else:
            # Assume JSON response that can be parsed
            json_data = await self.json()
            if hasattr(effective_cast_to, 'from_dict'):
                return effective_cast_to.from_dict(json_data)  # type: ignore
            elif hasattr(effective_cast_to, 'model_validate'):
                return effective_cast_to.model_validate(json_data)  # type: ignore
            else:
                return json_data  # type: ignore

    async def close(self) -> None:
        """
        Asynchronously close the response and release the connection.
        """
        await self._response.aclose()

    async def iter_bytes(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        """
        Async iterator for byte chunks of response content.

        :param chunk_size: Size of chunks to yield
        :return: Async iterator of byte chunks
        """
        async for chunk in self._response.aiter_bytes(chunk_size=chunk_size):
            yield chunk

    async def iter_text(self, chunk_size: int = 1024) -> AsyncIterator[str]:
        """
        Async iterator for text chunks of response content.

        :param chunk_size: Size of chunks to yield
        :return: Async iterator of text chunks
        """
        async for chunk in self._response.aiter_text(chunk_size=chunk_size):
            yield chunk

    async def iter_lines(self) -> AsyncIterator[str]:
        """
        Async iterator yielding line chunks of response content.

        :return: Async iterator of lines
        """
        async for line in self._response.aiter_lines():
            yield line

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Async context manager exit."""
        await self.close()
