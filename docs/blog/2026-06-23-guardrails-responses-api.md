---
slug: guardrails-responses-api
title: "Under the Hood: How OGX Enforces Guardrails Inside the Agentic Loop"
authors:
  - leseb
tags: [responses-api, guardrails, safety, streaming, agents]
date: 2026-06-23
---

AI agents that call tools, search documents, and reason over multiple turns are powerful, but they also need boundaries. A model that can execute a web search or query your internal knowledge base should not be free to produce harmful content along the way.

OGX implements guardrails as a first-class feature of the Responses API. Unlike bolt-on moderation that checks content after the fact, OGX validates content at two critical points inside the agentic loop: before inference starts and while user-visible text and reasoning output streams. This post explains exactly how that works, why the design choices matter, and how to use it in practice.

<!--truncate-->

## The problem with post-hoc moderation

Most moderation systems work outside the generation pipeline. You send a prompt to the model, get a response, then send that response to a moderation endpoint. If the content is flagged, you discard it and show an error.

This approach has two problems:

1. **Wasted compute.** The model generates the full response before you discover it violates a policy. For long, multi-turn agentic responses with tool calls, this can mean minutes of wasted work.
2. **Latency gap during streaming.** If you are streaming tokens to the user, you either block the entire stream until moderation completes (defeating the purpose of streaming) or you show tokens before they are validated (defeating the purpose of moderation).

OGX solves both problems by embedding guardrail checks directly inside the response orchestration loop.

## How to use guardrails

From the client side, guardrails are a single boolean on the Responses API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="unused")

response = client.responses.create(
    model="openai/gpt-4o-mini",
    input="Summarize this text.",
    extra_body={"guardrails": True},
)
print(response.output_text)
```

Setting `guardrails` to `True` tells the server to run request input plus generated text and reasoning content through the configured moderation endpoint. If the content is flagged, or if moderation cannot be completed safely, the response is replaced with a refusal. No shield IDs, no model lists, no moderation credentials on the client side.

The same parameter works with streaming, tool calls, MCP, and multi-turn conversations. No changes to your orchestration logic are needed.

### Why a boolean?

Earlier versions of OGX required clients to pass a list of shield identifiers (e.g., `"guardrails": ["llama-guard", "content-filter"]`), and the server would resolve each ID to a registered provider, manage routing tables, and coordinate multiple safety backends. This created real operational complexity: six safety providers with different auth, configuration, and edge cases, all for a feature that most deployments use as a simple yes/no gate.

The new design replaces all of that with server-side moderation configuration and a boolean on the client. The platform administrator picks the moderation service once; application developers just flip the switch.

## Server configuration

Guardrails require a `moderation_endpoint` to be configured on the builtin responses provider. This is the URL of any OpenAI-compatible `/v1/moderations` endpoint. If the endpoint requires authentication, configure `moderation_headers` beside it.

```yaml
providers:
  responses:
    - provider_id: builtin
      provider_type: inline::builtin
      config:
        moderation_endpoint: "https://api.openai.com/v1/moderations"
        moderation_headers:
          Authorization: "Bearer ${env.OPENAI_API_KEY}"
```

You can point this at OpenAI's moderation API, an OpenAI-compatible gateway in front of a hosted content safety service, a self-hosted moderation model, or any service that accepts `POST {"input": "text"}` and returns `{"results": [{"flagged": bool, "categories": {...}}]}`. The server makes a direct HTTP call — no proxy layer, no routing table, no provider abstraction in between.

`moderation_headers` are server-side only. They are never exposed to clients, and they are treated as sensitive configuration when OGX redacts config output. This keeps moderation credentials on the platform side and out of application requests.

If a client sends `guardrails: True` but no `moderation_endpoint` is configured, the server returns an error immediately rather than silently skipping validation:

```python
if enable_guardrails and not self.moderation_endpoint:
    raise ServiceNotEnabledError(
        "moderation_endpoint",
        provider_specific_message=(
            "Guardrails require a moderation endpoint to be configured "
            "on the server. Contact your platform administrator to set "
            "'moderation_endpoint' on the responses provider, or remove "
            "the 'guardrails' parameter from your request."
        ),
    )
```

## The agentic loop

To understand where guardrails fit, you need to understand how OGX orchestrates a Responses API call. The core of the implementation is the `StreamingResponseOrchestrator` class, which runs an iterative loop that interleaves inference, tool execution, and content validation.

Here is the high-level flow:

```text
1. Client sends request with input, tools, and guardrails: True
2. Server converts input to chat completion messages
3. INPUT GUARDRAIL CHECK ← validates all user messages
4. If violation → return refusal immediately
5. Enter agentic loop:
   a. Call inference (chat completion) with current messages
   b. Buffer generated text/reasoning chunks
      └── OUTPUT GUARDRAIL CHECK ← validates buffered content in batches
      └── If violation → replace stream with refusal
   c. Parse tool calls from model output
   d. Execute server-side tools (web search, file search, MCP)
   e. Append tool results to message history
   f. If more tool calls needed → go to (a)
   g. If only client-side function calls → return to client
6. Emit final response (completed / incomplete / failed)
```

The loop continues until one of these conditions is met:

- The model produces a final text response with no tool calls
- The maximum iteration count is reached (default: 10)
- Only client-side function calls remain (the client needs to execute them)
- A guardrail violation is detected
- The `max_output_tokens` budget is exhausted

Each iteration through the loop is a full inference call. The model sees the accumulated conversation history including previous tool results, so it can reason about what it has learned and decide what to do next.

## Checkpoint 1: input validation

Before the agentic loop begins, OGX checks the user's input against the moderation endpoint:

```python
if self.enable_guardrails:
    combined_text = interleaved_content_as_str(
        [msg.content for msg in self.ctx.messages]
    )
    input_violation_message = await run_guardrails(
        self.moderation_endpoint,
        combined_text,
        headers=self.moderation_headers,
    )
    if input_violation_message:
        yield await self._create_refusal_response(input_violation_message)
        return
```

This flattens all input messages into a single text string and sends it to the moderation endpoint. If the content is flagged, the entire response is short-circuited: no inference call happens, no tokens are generated, no tools are executed. The client receives a `response.completed` event containing a refusal content part instead of the model's output.

This matters because it prevents the model from ever seeing harmful input. Without input validation, a jailbreak prompt could manipulate the model into producing harmful tool calls or responses that might pass output validation in isolation.

## Checkpoint 2: batched streaming validation

Output validation is more nuanced. During streaming, the model generates tokens one at a time. Calling the moderation endpoint for every token would be prohibitively slow. But waiting for the entire response would defeat the purpose of streaming.

OGX takes a middle path: **batched chunk validation**. Here is how it works:

```text
For each streaming chunk from the inference provider:
  1. Accumulate text and reasoning content deltas
  2. Buffer the streaming event (don't emit to client yet)
  3. Track characters since last check

  When characters >= 200:
    a. Send accumulated text to moderation endpoint
    b. If violation:
       - Discard all buffered events
       - Emit refusal response
       - Stop processing the stream
    c. If clean:
       - Emit all buffered events to client
       - Reset character counter

  After stream ends:
    Final guardrail check on any remaining buffered content
```

The 200-character batch size is a deliberate tradeoff. Smaller batches catch violations sooner but increase moderation API calls. Larger batches reduce overhead but delay detection. 200 characters is roughly a sentence, which gives the moderation model enough context to make accurate judgments while keeping latency acceptable.

Here is the core of the batched validation logic:

```python
_GUARDRAIL_BATCH_CHARS = 200

# Inside _process_streaming_chunks:
# Reasoning characters count toward the batch threshold alongside text content.
guardrail_check_due = chars_since_last_check >= _GUARDRAIL_BATCH_CHARS

if self.enable_guardrails and guardrail_check_due:
    accumulated_text = "".join(chat_response_content + reasoning_text_accumulated)
    violation_message = await run_guardrails(
        self.moderation_endpoint,
        accumulated_text,
        headers=self.moderation_headers,
    )
    if violation_message:
        pending_guardrail_events.clear()
        yield await self._create_refusal_response(violation_message)
        self.violation_detected = True
        return
    for event in pending_guardrail_events:
        yield event
    pending_guardrail_events.clear()
    chars_since_last_check = 0
```

A key detail: each validation call sends the **entire accumulated text so far** — including both text content and reasoning content — not just the new batch. This gives the moderation model full context for content whose risk depends on previous sentences.

### What happens when reasoning is present

Some models produce reasoning content (chain-of-thought) alongside their text output. Reasoning events are user-visible in the Responses stream, so they must pass through guardrail validation just like text content.

When guardrails are enabled, reasoning events are buffered alongside text events and included in the accumulated text sent to the moderation endpoint. Reasoning characters count toward the 200-character batch threshold, so reasoning-only responses still trigger timely moderation checks. No reasoning content reaches the client until it has been validated.

### The final flush

After the inference stream ends, there may be buffered events that have not reached the 200-character threshold. OGX runs one final guardrail check on this remaining content before emitting it:

```python
if self.enable_guardrails and pending_guardrail_events:
    accumulated_text = "".join(chat_response_content + reasoning_text_accumulated)
    violation_message = await run_guardrails(
        self.moderation_endpoint,
        accumulated_text,
        headers=self.moderation_headers,
    )
    if violation_message:
        pending_guardrail_events.clear()
        yield await self._create_refusal_response(violation_message)
        self.violation_detected = True
        return
    for event in pending_guardrail_events:
        yield event
```

This ensures the final buffered text and reasoning events are validated before they reach the client, regardless of how the stream ends.

## The moderation call

The `run_guardrails` function is intentionally simple. It makes a single HTTP POST to the configured endpoint using the OpenAI moderation format:

```python
async def run_guardrails(
    moderation_endpoint: str | None,
    messages: str,
    headers: dict[str, str] | None = None,
) -> str | None:
    if not messages or not moderation_endpoint:
        return None

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            resp = await client.post(
                moderation_endpoint,
                json={"input": messages},
                headers=headers,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL):
            logger.warning(
                "Failed to call moderation endpoint", endpoint=moderation_endpoint
            )
            return "Failed to validate content: moderation service unavailable"

    try:
        data = resp.json()
    except Exception:
        logger.warning(
            "Failed to parse moderation response as JSON", endpoint=moderation_endpoint
        )
        return (
            "Failed to validate content: moderation service returned invalid response"
        )

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        logger.warning(
            "Moderation endpoint returned unexpected format",
            endpoint=moderation_endpoint,
        )
        return "Failed to validate content: moderation response has unexpected format"
    if not results:
        logger.warning(
            "Moderation endpoint returned no results", endpoint=moderation_endpoint
        )
        return "Failed to validate content: moderation response has unexpected format"

    for result in results:
        if not isinstance(result, dict):
            logger.warning(
                "Failed to parse moderation result entry", endpoint=moderation_endpoint
            )
            return (
                "Failed to validate content: moderation response has unexpected format"
            )
        flagged = result.get("flagged")
        if not isinstance(flagged, bool):
            logger.warning(
                "Failed to parse moderation result flagged field",
                endpoint=moderation_endpoint,
            )
            return (
                "Failed to validate content: moderation response has unexpected format"
            )
        categories = result.get("categories", {})
        if not isinstance(categories, dict):
            logger.warning(
                "Failed to parse moderation result categories",
                endpoint=moderation_endpoint,
            )
            return (
                "Failed to validate content: moderation response has unexpected format"
            )
        if flagged:
            flagged_cats = [c for c, f in categories.items() if f]
            msg = "Content blocked by safety guardrails"
            if flagged_cats:
                msg += f" (flagged for: {', '.join(flagged_cats)})"
            return msg

    return None
```

A key detail: this function **fails closed**. If the moderation service is unreachable, returns an error, or returns a malformed response, the function returns a blocking message rather than `None`. Content is never allowed through when the moderation check cannot be performed. Only a clean, parseable response with `"flagged": false` returns `None`.

No provider abstraction, no routing table, no shield resolution. One HTTP call, one JSON response. The moderation endpoint is expected to return the [OpenAI moderation response format](https://platform.openai.com/docs/api-reference/moderations/object): a `results` array where each element has `flagged` (boolean) and `categories` (dict of category names to booleans). The optional `headers` parameter lets the server pass authentication credentials configured via `moderation_headers` in the provider config.

This simplicity is a feature. The previous implementation required registering safety providers, managing shield routing tables, resolving guardrail IDs to model IDs, and coordinating multiple backends through an internal Safety API. All of that infrastructure existed to support a `/v1/moderations` proxy endpoint that clients could call directly — but the only place where server-side moderation actually adds value is inside the agentic loop, where the orchestrator needs to check content mid-generation. That is exactly what `run_guardrails` does.

## The refusal response

When a guardrail violation is detected at either checkpoint, OGX constructs a complete `response.completed` event with a refusal content part:

```python
async def _create_refusal_response(self, violation_message: str):
    refusal_content = OpenAIResponseContentPartRefusal(refusal=violation_message)
    refusal_response = OpenAIResponseObject(
        id=self.response_id,
        status="completed",
        output=[
            OpenAIResponseMessage(
                role="assistant",
                content=[refusal_content],
            )
        ],
        # ... other fields preserved
    )
    return OpenAIResponseObjectStreamResponseCompleted(response=refusal_response)
```

The refusal is a proper Responses API object. Clients that handle the `refusal` content type can display an appropriate message. The violation message includes which categories were flagged, so applications can take context-appropriate action.

## Guardrails and the multi-turn tool loop

Guardrails interact with the agentic loop in an important way: they check generated text and reasoning content, not raw tool results or tool-call arguments. Here is why.

Server-side tools (web search, file search, MCP) are executed by the server in a controlled environment. Their results are structured data that gets injected into the conversation history for the next inference call. The model then reasons about those results and produces text output, which is where guardrails apply.

This design means:

- A web search result containing harmful content will not trigger a guardrail violation on its own
- But if the model incorporates that harmful content into text or reasoning output, that output goes through moderation
- Tool calls themselves are not blocked by guardrails; tool authorization remains a separate control from content moderation

When a violation is detected mid-loop, the `violation_detected` flag stops all further processing:

```python
async for stream_event_or_result in self._process_streaming_chunks(
    completion_result, output_messages
):
    if isinstance(stream_event_or_result, ChatCompletionResult):
        completion_result_data = stream_event_or_result
    else:
        yield stream_event_or_result

# If violation detected, skip the rest of processing
if self.violation_detected:
    return
```

No further inference iterations are made, no more tools are executed, and no more buffered text or reasoning events are emitted.

## Design principles

Several principles guided this implementation:

**Fail closed, not open.** If guardrails are requested and no moderation endpoint is configured, the request fails rather than proceeding without validation.

**No unvalidated buffered content.** Text and reasoning events are held until the current guardrail batch passes. When a violation is detected during streaming, the current buffer is discarded, a refusal is emitted, and generation stops.

**Validation with context.** Each output guardrail check sends the full accumulated text, not just the latest batch. This helps catch content whose risk only becomes clear when multiple chunks are read together.

**Minimal latency impact.** Input validation adds one moderation call before inference. Output validation adds roughly one call per 200 characters of text and reasoning output, plus a final check for any remaining buffered content.

**Guardrails are opt-in.** When `guardrails` is not set, no moderation checks are performed, and streaming events flow directly to the client.

**Configuration belongs on the server.** The choice of moderation service is an infrastructure decision, not an application decision. Platform administrators set `moderation_endpoint` once; application developers just pass `guardrails: True`.

## What was removed and why

The previous guardrails implementation was built on top of a full Safety API subsystem: protocol definitions, routing tables, a shield registry, seven safety providers (Llama Guard, Prompt Guard, Code Scanner, Bedrock, NVIDIA, SambaNova, Passthrough), and a standalone `/v1/moderations` proxy endpoint.

All of that has been removed. The standalone `/v1/moderations` endpoint added a network hop for zero value — clients already know how to call moderation services directly. The provider abstraction existed to support that proxy, but the only server-side value is guardrails during generation, where the orchestrator needs to run moderation checks mid-stream. That is now a direct HTTP call from the responses provider.

The result is less code, fewer moving parts, and a clearer contract: one endpoint, optional server-side headers, one HTTP call per check, and one boolean flag on the request.

## Try it

If you are running OGX, you can add guardrails to any existing Responses API call:

```python
response = client.responses.create(
    model="openai/gpt-4o-mini",
    input="Your prompt here",
    tools=[{"type": "web_search_preview"}],
    extra_body={"guardrails": True},
)
```

Guardrails work inside the same Responses API loop that handles streaming, tool calling, MCP, file search, multi-turn conversations, and reasoning models. Text and reasoning content pass through the same moderation path regardless of which features the response uses.

For more details on the implementation, see the source in [`streaming.py`](https://github.com/ogx-ai/ogx/blob/main/src/ogx/providers/inline/responses/builtin/responses/streaming.py) and [`utils.py`](https://github.com/ogx-ai/ogx/blob/main/src/ogx/providers/inline/responses/builtin/responses/utils.py).
