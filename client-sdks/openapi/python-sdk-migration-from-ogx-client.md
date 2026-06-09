# Migrating from `ogx_client` to `ogx_open_client`

This document describes all user-facing changes when migrating from the
Stainless-generated `ogx_client` SDK to the new `ogx_open_client` SDK.

---

## TL;DR

For most codebases, this migration is **a search-and-replace**. The API methods
accept the same keyword arguments, list responses are still iterable, streaming
works the same way, and exceptions have the same names.

The two things you will definitely do:

1. **Replace the package name** in every import:
   `ogx_client` -> `ogx_open_client`
2. **Replace the type import path and names**:
   `from ogx_client.types import ResponseObject` -> `from ogx_open_client.models import OpenAIResponseObject`

Everything else in this document covers edge cases and advanced usage that most
code does not touch. The sections below are ordered by how likely they are to
affect you.

---

## Table of Contents

**Most codebases (search-and-replace):**

1. [Package Name and Imports](#1-package-name-and-imports)
2. [Type and Model Name Changes](#2-type-and-model-name-changes)

**Only if you use these specific features:**

1. [Environment Variables and Authentication](#3-environment-variables-and-authentication)
2. [List and Pagination Responses](#4-list-and-pagination-responses)
3. [Removed Client Features](#5-removed-client-features)
4. [Exception Attribute Changes](#6-exception-attribute-changes)

**Unlikely to affect you (edge cases and internals):**

1. [Client Initialization (new options)](#7-client-initialization)
2. [API Method Calling Convention](#8-api-method-calling-convention)
3. [Streaming Internals](#9-streaming-internals)
4. [APIResponse Changes](#10-apiresponse-changes)
5. [Params Types Replaced by Request Models](#11-params-types-replaced-by-request-models)
6. [Async Client Changes](#12-async-client-changes)
7. [Dependencies and Python Version](#13-dependencies-and-python-version)

---

## 1. Package Name and Imports

Every import changes. This is a straightforward find-and-replace.

```python
# Before
from ogx_client import OgxClient
from ogx_client.types import ResponseObject

# After
from ogx_open_client import OgxClient
from ogx_open_client.models import OpenAIResponseObject
```

The `types` sub-package is now called `models`. All models live in a single flat
namespace -- there are no sub-packages like `types.shared`, `types.alpha`,
`types.chat`, etc.

```python
# Before -- types scattered across sub-packages
from ogx_client.types import ResponseObject
from ogx_client.types.shared import HealthInfo
from ogx_client.types.alpha import InferenceRerankResponse
from ogx_client.types.chat import CompletionCreateResponse
from ogx_client.types.vector_stores import VectorStoreFile

# After -- everything from one place
from ogx_open_client.models import (
    OpenAIResponseObject,
    HealthInfo,
    RerankResponse,
    OpenAIChatCompletion,
    VectorStoreFileObject,
)
```

The `Client` and `AsyncClient` short aliases no longer exist. Use `OgxClient`
and `AsyncOgxClient`.

---

## 2. Type and Model Name Changes

Many types have been renamed. The actual data and fields on these types are
the same -- only the class names changed. Here are the patterns:

- **`OpenAI` prefix added** to types that wrap OpenAI-compatible API objects
- **`Object` suffix added** to resource types (e.g., `VectorStore` -> `VectorStoreObject`)
- **Per-operation wrappers collapsed** (e.g., `BatchCreateResponse` / `BatchRetrieveResponse` / `BatchCancelResponse` all become `Batch`)

### Core Resource Types

| `ogx_client.types.*` | `ogx_open_client.models.*` |
|----------------------|---------------------------|
| `File` | `OpenAIFileObject` |
| `Model` | `OpenAIModel` |
| `Prompt` | `Prompt` |
| `VectorStore` | `VectorStoreObject` |
| `ResponseObject` | `OpenAIResponseObject` |
| `ResponseMessage` | `OpenAIResponseMessage` |
| `CompactedResponse` | `OpenAICompactedResponse` |
| `ResponseObjectStream` | `OpenAIResponseObjectStream` |
| `ConversationObject` | `Conversation` |
| `ChatCompletionChunk` | `OpenAIChatCompletionChunk` |
| `CompletionCreateResponse` | `OpenAICompletion` |
| `CreateEmbeddingsResponse` | `OpenAIEmbeddingsResponse` |
| `QueryChunksResponse` | `QueryChunksResponse` |

### Delete Response Types

| `ogx_client.types.*` | `ogx_open_client.models.*` |
|----------------------|---------------------------|
| `DeleteFileResponse` | `OpenAIFileDeleteResponse` |
| `ResponseDeleteResponse` | `OpenAIDeleteResponseObject` |
| `ConversationDeleteResponse` | `ConversationDeletedResource` |
| `VectorStoreDeleteResponse` | `VectorStoreDeleteResponse` |
| `VectorStoreSearchResponse` | `VectorStoreSearchResponse` |

### Batch Types (collapsed)

| `ogx_client.types.*` | `ogx_open_client.models.*` |
|----------------------|---------------------------|
| `BatchCreateResponse` | `Batch` |
| `BatchRetrieveResponse` | `Batch` |
| `BatchListResponse` | `Batch` |
| `BatchCancelResponse` | `Batch` |

### List Response Types

| `ogx_client.types.*` | `ogx_open_client.models.*` |
|----------------------|---------------------------|
| `ModelListResponse` | `ListModelsV1ModelsGet200Response` |
| `ModelRetrieveResponse` | `GetModelV1ModelsModelIdGet200Response` |
| `ListModelsResponse` | `ListModelsResponse` |
| `ListFilesResponse` | `ListOpenAIFileResponse` |
| `ListPromptsResponse` | `ListPromptsResponse` |
| `ListVectorStoresResponse` | `VectorStoreListResponse` |
| `ResponseListResponse` | `OpenAIResponseObjectWithInput` |
| `PromptListResponse` | `ListPromptsResponse` |
| `ProviderListResponse` | `ListProvidersResponse` |
| `RouteListResponse` | `ListRoutesResponse` |

### Shared Types (moved from `types.shared` to flat `models`)

| `ogx_client.types.shared.*` | `ogx_open_client.models.*` |
|-----------------------------|---------------------------|
| `ParamType` | `ParamType` |
| `RouteInfo` | `RouteInfo` |
| `HealthInfo` | `HealthInfo` |
| `VersionInfo` | `VersionInfo` |
| `ProviderInfo` | `ProviderInfo` |
| `SystemMessage` | `SystemMessage` |
| `SamplingParams` | `SamplingParams` |
| `InterleavedContent` | `InterleavedContent` |
| `InterleavedContentItem` | `InterleavedContentItem` |
| `ListRoutesResponse` | `ListRoutesResponse` |
| `ListProvidersResponse` | `ListProvidersResponse` |

### Alpha Types

| `ogx_client.types.alpha.*` | `ogx_open_client.models.*` |
|----------------------------|---------------------------|
| `InferenceRerankResponse` | `RerankResponse` |
| `InferenceRerankParams` | `RerankRequest` |
| `AdminListRoutesParams` | `ListRoutesRequest` |

### Chat Types

| `ogx_client.types.chat.*` | `ogx_open_client.models.*` |
|---------------------------|---------------------------|
| `CompletionCreateResponse` | `OpenAIChatCompletion` |
| `CompletionRetrieveResponse` | `OpenAIChatCompletion` |
| `CompletionListResponse` | `ListOpenAIChatCompletionResponse` |
| `CompletionCreateParams` | `OpenAIChatCompletionRequestWithExtraBody` |
| `CompletionListParams` | `ListChatCompletionsRequest` |

### Chat Completions Sub-types

| `ogx_client.types.chat.completions.*` | `ogx_open_client.models.*` |
|---------------------------------------|---------------------------|
| `MessageListResponse` | `ChatCompletionMessage` |
| `MessageListParams` | `ListChatCompletionMessagesRequest` |

### Conversations Types

| `ogx_client.types.conversations.*` | `ogx_open_client.models.*` |
|------------------------------------|---------------------------|
| `ItemGetResponse` | `ConversationItem` |
| `ItemCreateResponse` | `ConversationItem` |
| `ItemListResponse` | `ConversationItemList` |
| `ItemCreateParams` | `ConversationItemCreateRequest` |
| `ItemListParams` | `ListItemsRequest` |
| `ItemGetParams` | `RetrieveItemRequest` |

### Models Sub-types

| `ogx_client.types.models.*` | `ogx_open_client.models.*` |
|-----------------------------|---------------------------|
| `OpenAIListResponse` | `OpenAIListModelsResponse` |

### Responses Sub-types

| `ogx_client.types.responses.*` | `ogx_open_client.models.*` |
|--------------------------------|---------------------------|
| `InputItemListResponse` | `ListOpenAIResponseInputItem` |

### Vector Stores Sub-types

| `ogx_client.types.vector_stores.*` | `ogx_open_client.models.*` |
|------------------------------------|---------------------------|
| `VectorStoreFile` | `VectorStoreFileObject` |
| `VectorStoreFileBatches` | `VectorStoreFileBatchObject` |
| `FileDeleteResponse` | `VectorStoreFileDeleteResponse` |
| `FileContentResponse` | `VectorStoreFileContentResponse` |
| `ListVectorStoreFilesInBatchResponse` | `VectorStoreFilesListInBatchResponse` |
| `FileCreateParams` | `OpenAIAttachFileRequest` |
| `FileUpdateParams` | `OpenAIUpdateVectorStoreFileRequest` |
| `FileBatchCreateParams` | `OpenAICreateVectorStoreFileBatchRequestWithExtraBody` |

---

## 3. Environment Variables and Authentication

**Affects you if** your code uses `OGX_CLIENT_API_KEY`, `OGX_CLIENT_BASE_URL`,
or `OGX_CLIENT_CUSTOM_HEADERS` environment variables, **or** if you pass
`api_key=` to the client constructor.

### Environment variables no longer read

`ogx_client` auto-read three environment variables. `ogx_open_client` reads
none of them:

| Environment Variable | `ogx_client` | `ogx_open_client` |
|---------------------|-------------|-------------------|
| `OGX_CLIENT_API_KEY` | Auto-inferred for `api_key` | **Not read** |
| `OGX_CLIENT_BASE_URL` | Auto-inferred for `base_url` | **Not read** |
| `OGX_CLIENT_CUSTOM_HEADERS` | Parsed and merged into default headers | **Not read** |

- **`OGX_CLIENT_BASE_URL`**: pass `base_url=` explicitly instead.
- **`OGX_CLIENT_CUSTOM_HEADERS`**: pass `default_headers=` explicitly instead
  (accepts a `dict`, not the newline-separated string format the env var used).
- **`OGX_CLIENT_API_KEY`**: see the gotcha below.

If you need to keep using env vars, read them yourself:

```python
import os
from ogx_open_client import OgxClient

client = OgxClient(
    base_url=os.environ.get("OGX_CLIENT_BASE_URL", "http://localhost:8321"),
    default_headers={"Authorization": f"Bearer {os.environ['OGX_CLIENT_API_KEY']}"},
)
```

### `api_key=` is accepted but silently ignored

**This is the most dangerous gotcha in the migration.** The `api_key` parameter
still exists on the `OgxClient` constructor, so this code **does not raise an
error**:

```python
# Looks correct, but the API key is NEVER sent to the server
client = OgxClient(base_url="http://localhost:8321", api_key="my-secret-key")
```

In `ogx_client`, this would automatically send
`Authorization: Bearer my-secret-key` with every request. In `ogx_open_client`,
the key is stored on `self.api_key` but **never added to any request header**.
Your requests will go out unauthenticated, and depending on your server
configuration you may get silent 401 errors or unexpected behavior.

**The fix:** use `default_headers` to send the API key explicitly:

```python
from ogx_open_client import OgxClient

api_key = "my-secret-key"
client = OgxClient(
    base_url="http://localhost:8321",
    default_headers={"Authorization": f"Bearer {api_key}"},
)
```

This applies whether you were passing `api_key=` directly or relying on the
`OGX_CLIENT_API_KEY` environment variable.

If you were connecting to a server that does not require authentication (common
in local development), this does not affect you.

---

## 4. List and Pagination Responses

**Only affects you if** you were relying on `SyncOpenAICursorPage`'s
auto-pagination to transparently fetch all pages across multiple API calls.

### What still works

Basic iteration over list results works identically:

```python
# This works in both SDKs
for file in client.files.list():
    print(file.id)
```

The new list response objects implement `__iter__`, `__getitem__`, and `__len__`,
all delegating to their internal `.data` list. So `for item in result`,
`result[0]`, and `len(result)` all work.

### What changed

`ogx_client` used `SyncOpenAICursorPage[T]` for some endpoints -- an
auto-paginating iterator that transparently made additional API calls when you
iterated past the current page. `ogx_open_client` returns one page at a time.
If you have more data than fits in one page, you need to paginate manually:

```python
result = client.files.list()
# process result...
while result.has_more:
    result = client.files.list(after=result.last_id)
    # process next page...
```

In practice, most code fetches a single page and does not rely on
auto-pagination. If your lists are small enough to fit in one response (the
common case), the behavior is identical.

### Return type changes per endpoint

| Method | `ogx_client` return | `ogx_open_client` return |
|--------|--------------------|-----------------------|
| `responses.list()` | `SyncOpenAICursorPage[ResponseListResponse]` | `ListOpenAIResponseObject` |
| `responses.create()` | `ResponseObject` | `OpenAIResponseObject` |
| `files.list()` | `SyncOpenAICursorPage[File]` | `ListOpenAIFileResponse` |
| `batches.list()` | `SyncOpenAICursorPage[BatchListResponse]` | `ListBatchesResponse` |
| `vector_stores.list()` | `SyncOpenAICursorPage[VectorStore]` | `VectorStoreListResponse` |
| `vector_stores.files.list()` | `SyncOpenAICursorPage[VectorStoreFile]` | `VectorStoreListFilesResponse` |
| `conversations.items.list()` | `SyncOpenAICursorPage[ItemListResponse]` | `ConversationItemList` |
| `chat.completions.messages.list()` | `SyncOpenAICursorPage[MessageListResponse]` | `ChatCompletionMessageList` |
| `models.list()` | `ModelListResponse` (TypeAlias Union) | `ListModelsV1ModelsGet200Response` (OneOf wrapper) |
| `chat.completions.list()` | `CompletionListResponse` | `ListOpenAIChatCompletionResponse` |
| `providers.list()` | `ProviderListResponse` (`TypeAlias = List[ProviderInfo]`) | `ListProvidersResponse` |
| `routes.list()` | `RouteListResponse` (`TypeAlias = List[RouteInfo]`) | `ListRoutesResponse` |
| `prompts.list()` | `PromptListResponse` (`TypeAlias = List[Prompt]`) | `ListPromptsResponse` |

### TypeAlias lists replaced by wrapper objects

A few endpoints in `ogx_client` returned plain Python lists via TypeAliases
(e.g., `ProviderListResponse = List[ProviderInfo]`). These are now Pydantic
models with a `.data` field. If you were treating the result as a bare list,
access `.data` or iterate (which delegates to `.data`):

```python
# Before -- was a plain list
providers = client.providers.list()  # List[ProviderInfo]
first = providers[0]

# After -- wrapper object, but iteration/indexing still works
providers = client.providers.list()  # ListProvidersResponse
first = providers[0]  # works via __getitem__
first = providers.data[0]  # also works
```

---

## 5. Removed Client Features

**Only affects you if** you used these specific patterns.

### `client.copy()` / `client.with_options()`

No longer available. Create a new client instead.

```python
# Before
new_client = client.copy(base_url="http://other-server:8321")

# After
new_client = OgxClient(base_url="http://other-server:8321")
```

### `client.with_raw_response`

No longer available. Use `*_with_http_info()` method variants:

```python
# Before
raw = client.with_raw_response.responses.retrieve("resp-123")
raw.status_code
raw.headers
parsed = raw.parse()

# After
resp = client.responses.retrieve_with_http_info("resp-123")
resp.status_code
resp.headers
resp.data  # already parsed
```

### `client.with_streaming_response`

No longer available. Use `*_without_preload_content()` method variants:

```python
# Before
with client.with_streaming_response.files.content("file-abc") as response:
    for chunk in response.iter_bytes():
        ...

# After
response = client.files.content_without_preload_content("file-abc")
```

### `file_from_path` utility

No longer available. Use standard Python file I/O:

```python
# Before
from ogx_client import file_from_path

# After
with open("path/to/file", "rb") as f:
    client.files.create(file=f, purpose="assistants")
```

### `aiohttp` backend

The optional `[aiohttp]` backend for `AsyncOgxClient` is no longer available.
`httpx.AsyncClient` is used exclusively.

---

## 6. Exception Attribute Changes

**Only affects you if** you catch SDK exceptions and inspect their attributes
(status code, response body, etc.). If you only catch them by type (e.g.,
`except NotFoundError:`), nothing changes -- the class names are the same.

### Catching by type still works

```python
# Works identically in both SDKs
from ogx_open_client import NotFoundError, BadRequestError, RateLimitError

try:
    client.models.retrieve("nonexistent")
except NotFoundError:
    print("not found")
```

### Attribute names changed

| What you want | `ogx_client` | `ogx_open_client` |
|---------------|-------------|-------------------|
| HTTP status code | `e.status_code` | `e.status` (also available as `e.status_code`) |
| Full HTTP response | `e.response` (`httpx.Response`) | Not available (individual fields below) |
| Response headers | `e.response.headers` | `e.headers` |
| Response body | `e.body` (parsed JSON or None) | `e.body` (raw string) |
| Error message | `e.message` | `e.message` (also available as `e.reason`) |
| HTTP request | `e.request` | Not available |

```python
try:
    client.models.retrieve("nonexistent")
# Before
except NotFoundError as e:
    print(e.status_code)  # 404
    print(e.response)  # httpx.Response

# After
except NotFoundError as e:
    print(e.status)  # 404
    print(e.body)  # raw body string
    print(e.headers)  # response headers dict
```

### Exception hierarchy

`ogx_client` had one hierarchy rooted at `OgxClientError -> APIError`.
`ogx_open_client` has two coexisting hierarchies:

**Primary (used by default):**

```text
OpenApiException
  -> ApiException -> BadRequestException, NotFoundException, ...
```

The familiar names (`BadRequestError`, `NotFoundError`, etc.) are **aliases**
for the `*Exception` classes. `from ogx_open_client import BadRequestError`
gives you `BadRequestException` under the hood.

**Secondary (Stainless-compatible, also importable):**

```text
OgxClientError -> APIError -> APIStatusError -> BadRequestError, ...
```

If you were catching `APIStatusError` or `OgxClientError` as a base class,
note that the primary hierarchy uses `ApiException` / `OpenApiException` instead.

---

## 7. Client Initialization

**This is mostly backward-compatible.** `base_url=` and `api_key=` still work.

The new SDK adds a `configuration` parameter as an alternative way to pass a URL
string or a full `Configuration` object:

```python
from ogx_open_client import OgxClient, Configuration

# All three are equivalent:
client = OgxClient(base_url="http://localhost:8321")
client = OgxClient(configuration="http://localhost:8321")
client = OgxClient(configuration=Configuration(host="http://localhost:8321"))

# Configuration gives access to more options:
config = Configuration(host="http://localhost:8321", timeout=30, retries=3, http2=True)
client = OgxClient(configuration=config)
```

### Constructor parameter comparison

| Parameter | `ogx_client` | `ogx_open_client` |
|-----------|-------------|-------------------|
| `base_url` | Direct param | Supported via `**kwargs` (forwarded to `Configuration`) |
| `api_key` | Direct param, auto-read from env | Direct param, **not** read from env |
| `timeout` | Direct param (`float \| Timeout`) | Via `**kwargs` or `Configuration(timeout=...)` |
| `max_retries` | Direct param (default: 2) | Via `**kwargs` or `Configuration(retries=...)` |
| `http_client` | Direct param (`httpx.Client`) | Not supported |
| `default_query` | Direct param | Not supported |
| `configuration` | Not supported | **New** -- accepts `Configuration` object or string URL |
| `header_name`/`header_value` | Not supported | Supported (single-header auth) |
| `cookie` | Not supported | Supported |

---

## 8. API Method Calling Convention

**This is backward-compatible for the common case.** If you call methods with
keyword arguments (the normal way), nothing changes:

```python
# Works identically in both SDKs
response = client.responses.create(model="gpt-4o", input="Hello, world!")
```

The new SDK additionally accepts a request body object as the first positional
argument:

```python
# New option (not available in ogx_client)
from ogx_open_client.models import CreateResponseRequest

request = CreateResponseRequest(model="gpt-4o", input="Hello, world!")
response = client.responses.create(request)
```

### Minor differences (unlikely to matter)

- `ogx_client` enforced keyword-only arguments via `*`. The new SDK does not --
  but since nobody passes positional args to these methods, this is invisible.
- `ogx_client` used an `omit` sentinel to distinguish "not provided" from
  `None`. The new SDK uses `None` as the default. In practice the server
  treats both the same way.
- Both SDKs support `extra_body` for sending additional fields.

---

## 9. Streaming Internals

**The usage pattern is identical.** Only internal details changed.

```python
# Works identically in both SDKs
stream = client.responses.create(model="gpt-4o", input="Hello", stream=True)
for event in stream:
    print(event)
```

The stream event type names changed (same rename pattern as other types):

| `ogx_client` | `ogx_open_client` |
|-------------|-------------------|
| `ResponseObjectStream` | `OpenAIResponseObjectStream` |
| `ChatCompletionChunk` | `OpenAIChatCompletionChunk` |

Other differences that only matter if you inspect stream internals:

| Aspect | `ogx_client` | `ogx_open_client` |
|--------|-------------|-------------------|
| Error in stream | Raises `APIError` | Silently caught |
| `response` attribute | `stream.response` (public) | `stream._response` (private) |
| `status_code` | `stream.response.status_code` | `stream.status_code` |
| `headers` | `stream.response.headers` | `stream.headers` |
| `until_done()` | Not available | Available |

---

## 10. APIResponse Changes

**Only affects you if** you used `client.with_raw_response` (see
[Section 5](#5-removed-client-features)).

`ogx_client`'s `APIResponse` was a wrapper around a live `httpx.Response` with
lazy parsing via `.parse()`.

`ogx_open_client`'s `ApiResponse` is a **Pydantic BaseModel** with the data
already parsed:

```python
# After (via *_with_http_info methods)
resp = client.responses.create_with_http_info(...)
resp.status_code  # int
resp.headers  # Mapping[str, str]
resp.data  # already-parsed model
resp.raw_data  # raw bytes
```

---

## 11. Params Types Replaced by Request Models

**Only affects you if** you explicitly imported `*Params` TypedDict types for
type annotations or static analysis.

`ogx_client` provided TypedDict classes (named `*Params`) for type-checking
method arguments. Most users never imported these -- they just passed keyword
arguments directly.

`ogx_open_client` replaces them with Pydantic `BaseModel` classes (named
`*Request`). If you referenced Params types in your code:

```python
# Before
from ogx_client.types import ResponseCreateParams


def my_func(params: ResponseCreateParams) -> None: ...


# After
from ogx_open_client.models import CreateResponseRequest


def my_func(params: CreateResponseRequest) -> None: ...
```

Full mapping:

| `ogx_client` (TypedDict) | `ogx_open_client` (BaseModel) |
|--------------------------|-------------------------------|
| `ResponseCreateParams` | `CreateResponseRequest` |
| `ResponseCompactParams` | `CompactResponseRequest` |
| `EmbeddingCreateParams` | `OpenAIEmbeddingsRequestWithExtraBody` |
| `CompletionCreateParams` (top-level) | `OpenAICompletionRequestWithExtraBody` |
| `chat.CompletionCreateParams` | `OpenAIChatCompletionRequestWithExtraBody` |
| `BatchCreateParams` | `CreateBatchRequest` |
| `BatchListParams` | `ListBatchesRequest` |
| `FileCreateParams` | `UploadFileRequest` |
| `FileListParams` | `ListFilesRequest` |
| `VectorStoreCreateParams` | `OpenAICreateVectorStoreRequestWithExtraBody` |
| `VectorStoreUpdateParams` | `OpenAIUpdateVectorStoreRequest` |
| `VectorStoreSearchParams` | `OpenAISearchVectorStoreRequest` |
| `VectorIoInsertParams` | `InsertChunksRequest` |
| `VectorIoQueryParams` | `QueryChunksRequest` |
| `ConversationCreateParams` | `CreateConversationRequest` |
| `ConversationUpdateParams` | `UpdateConversationRequest` |
| `PromptCreateParams` | `CreatePromptRequest` |
| `PromptUpdateParams` | `UpdatePromptRequest` |
| `PromptRetrieveParams` | `GetPromptRequest` |
| `PromptSetDefaultVersionParams` | `SetDefaultVersionRequest` |
| `alpha.InferenceRerankParams` | `RerankRequest` |
| `alpha.AdminListRoutesParams` | `ListRoutesRequest` |
| `vector_stores.FileCreateParams` | `OpenAIAttachFileRequest` |
| `vector_stores.FileUpdateParams` | `OpenAIUpdateVectorStoreFileRequest` |
| `vector_stores.FileBatchCreateParams` | `OpenAICreateVectorStoreFileBatchRequestWithExtraBody` |
| `conversations.ItemCreateParams` | `ConversationItemCreateRequest` |
| `conversations.ItemListParams` | `ListItemsRequest` |
| `conversations.ItemGetParams` | `RetrieveItemRequest` |
| `chat.CompletionListParams` | `ListChatCompletionsRequest` |
| `chat.completions.MessageListParams` | `ListChatCompletionMessagesRequest` |

---

## 12. Async Client Changes

**Backward-compatible.** Usage is the same:

```python
from ogx_open_client import AsyncOgxClient

async_client = AsyncOgxClient(base_url="http://localhost:8321")
response = await async_client.responses.create(model="gpt-4o", input="Hello")
```

The `Client` and `AsyncClient` short aliases are gone -- use `OgxClient` and
`AsyncOgxClient`.

The optional `aiohttp` backend is no longer available.
`httpx.AsyncClient` is used exclusively.

---

## 13. Dependencies and Python Version

| Dependency | `ogx_client` | `ogx_open_client` |
|-----------|-------------|-------------------|
| Python | >= 3.9 | **>= 3.12** |
| `httpx` | >= 0.23.0, < 1 | >= 0.28.1 |
| `pydantic` | >= 1.9.0, < 3 | **>= 2** |
| `anyio` | >= 3.5.0, < 5 | Not required |
| `distro` | >= 1.7.0, < 2 | Not required |
| `sniffio` | Required | Not required |
| `tqdm` | Required | Not required |
| `typing-extensions` | >= 4.14, < 5 | >= 4.7.1 |
| `python-dateutil` | Not required | >= 2.8.2 |

If you are on Python 3.9--3.11, you must upgrade to 3.12+. If you are on
Pydantic v1, you must upgrade to v2. Both SDKs use **httpx** as the HTTP
client.

---

## Quick Migration Checklist

**Everyone:**

- [ ] Upgrade Python to 3.12+ and Pydantic to 2.x (if not already)
- [ ] Replace `ogx_client` with `ogx_open_client` in `pip install` / `pyproject.toml`
- [ ] Find-and-replace `from ogx_client` -> `from ogx_open_client` in all imports
- [ ] Replace `from ogx_client.types` -> `from ogx_open_client.models`
- [ ] Update type names per the [mapping tables](#2-type-and-model-name-changes)

**If applicable:**

- [ ] **If you pass `api_key=`**: this is now silently ignored -- switch to
      `default_headers={"Authorization": f"Bearer {key}"}` instead
- [ ] Read env vars explicitly if you relied on `OGX_CLIENT_API_KEY` / `OGX_CLIENT_BASE_URL` / `OGX_CLIENT_CUSTOM_HEADERS`
- [ ] Replace `client.with_raw_response` with `*_with_http_info()` methods
- [ ] Replace `client.copy()` / `client.with_options()` with new client creation
- [ ] Add manual pagination if you relied on `SyncOpenAICursorPage` auto-pagination
- [ ] Update `e.status_code` -> `e.status` if you inspect exception attributes
