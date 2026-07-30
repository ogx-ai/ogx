# OpenResponses Conformance Gaps

Tracked by: <https://github.com/ogx-ai/ogx/issues/4818>

All 6 conformance tests currently fail at the **Zod schema validation step** —
`responseResourceSchema.safeParse(rawData)` rejects the response before semantic
validators (`hasOutput`, `completedStatus`, etc.) can run. The issues fall into
three groups.

---

## Group 1: Fields missing entirely from `OpenAIResponseObject`

These fields don't exist in
`src/ogx_api/openai_responses.py::OpenAIResponseObject` at all.
The OpenResponses spec requires them as non-nullable numbers.

| Field | Spec (Zod) | Fix |
|---|---|---|
| `presence_penalty` | `z.number()` | Add `presence_penalty: float = 0.0` |
| `frequency_penalty` | `z.number()` | Add `frequency_penalty: float = 0.0` |
| `top_logprobs` | `z.number().int()` | Add `top_logprobs: int = 0` |

**File:** `src/ogx_api/openai_responses.py`

---

## Group 2: Optional in ogx, required (non-nullable) in the spec

These fields exist in `OpenAIResponseObject` but are typed `| None = None`.
When no value is provided they serialize as `null`, which the Zod schema rejects
(no null branch in any of these).

| Field | Spec (Zod) | ogx today | Fix |
|---|---|---|---|
| `temperature` | `z.number()` | `float \| None = None` | Default to `1.0` |
| `top_p` | `z.number()` | `float \| None = None` | Default to `1.0` |
| `tool_choice` | union of 3 types, no null | `... \| None = None` | Default to `"auto"` |
| `truncation` | `z.enum(["auto","disabled"])`, no null | `str \| None = None` | Default to `"disabled"` |
| `background` | `z.boolean()` | `bool \| None = Field(default=None)` | Default to `False` |
| `service_tier` | `z.string()` | `str \| None = None` | Default to `"default"` |
| `tools` | `z.array(...)`, no null | `Sequence[...] \| None = None` | Default to `[]` |

Additionally, `_snapshot_response()` in
`src/ogx/providers/inline/agents/builtin/responses/streaming.py`
never forwards `temperature` or `top_p` to the response object even though they
are available on the context (`self.ctx.temperature`, `self.ctx.top_p`).

**Files:**

- `src/ogx_api/openai_responses.py`
- `src/ogx/providers/inline/agents/builtin/responses/streaming.py`

---

## Group 3: Output item sub-object issues

These are hit for each item in `output[]` once the top-level fields are fixed.

### `messageSchema` (`type: "message"` output items)

`OpenAIResponseMessage` in `src/ogx_api/openai_responses.py`:

| Field | Spec (Zod) | ogx today | Fix |
|---|---|---|---|
| `id` | `z.string()` (required) | `str \| None = None` | Ensure always populated |
| `status` | `z.enum(["in_progress","completed","incomplete"])` (required) | `str \| None = None` | Ensure always populated |

### `outputTextContentSchema` (content parts inside messages)

`OpenAIResponseOutputMessageContentOutputText`:

| Field | Spec (Zod) | ogx today | Fix |
|---|---|---|---|
| `logprobs` | `z.array(...)` (required, non-nullable) | `list \| None = None` | Default to `[]` |

### `functionCallSchema` (`type: "function_call"` output items)

Relevant to the **Tool Calling** test. `OpenAIResponseFunctionCall`:

| Field | Spec (Zod) | ogx today | Fix |
|---|---|---|---|
| `id` | `z.string()` (required) | `str \| None = None` | Ensure always populated |
| `status` | required | `str \| None = None` | Ensure always populated |

**File:** `src/ogx_api/openai_responses.py`

---

## Summary

| # | Location | Change |
|---|---|---|
| 1 | `OpenAIResponseObject` | Add `presence_penalty: float = 0.0` |
| 2 | `OpenAIResponseObject` | Add `frequency_penalty: float = 0.0` |
| 3 | `OpenAIResponseObject` | Add `top_logprobs: int = 0` |
| 4 | `OpenAIResponseObject` | `temperature` default `1.0` (never None) |
| 5 | `OpenAIResponseObject` | `top_p` default `1.0` (never None) |
| 6 | `OpenAIResponseObject` | `tool_choice` default `"auto"` (never None) |
| 7 | `OpenAIResponseObject` | `truncation` default `"disabled"` (never None) |
| 8 | `OpenAIResponseObject` | `background` default `False` (never None) |
| 9 | `OpenAIResponseObject` | `service_tier` default `"default"` (never None) |
| 10 | `OpenAIResponseObject` | `tools` default `[]` (never None) |
| 11 | `streaming.py` `_snapshot_response()` | Forward `temperature` and `top_p` from context |
| 12 | `OpenAIResponseMessage` | `id` and `status` always populated |
| 13 | `OpenAIResponseOutputMessageContentOutputText` | `logprobs` default `[]` (never None) |
| 14 | `OpenAIResponseFunctionCall` | `id` and `status` always populated |
