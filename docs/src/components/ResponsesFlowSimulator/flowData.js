/**
 * Flow data for the Responses API sequence diagram simulator.
 *
 * Actors, flow steps, presets, validations, and prose fragments are defined
 * here and consumed by the React component.
 */

// ---------------------------------------------------------------------------
// Actors
// ---------------------------------------------------------------------------

export const ACTORS = {
  client:        { id: 'client',        label: 'Client',        description: 'Your application making the API call' },
  fastapi:       { id: 'fastapi',       label: 'FastAPI',       description: 'HTTP endpoint handler — routes, SSE, request validation' },
  impl:          { id: 'impl',          label: 'Responses',     description: 'Core orchestration — request routing, state management, conversation sync' },
  bgworker:      { id: 'bgworker',      label: 'BGWorker',      description: 'Async worker pool (10 workers) for background response processing' },
  orchestrator:  { id: 'orchestrator',  label: 'Orchestrator',  description: 'Streaming response orchestrator — runs the inference loop and coordinates tool execution' },
  safety:        { id: 'safety',        label: 'Safety',        description: 'Input/output guardrail validation via configured safety policies' },
  inference:     { id: 'inference',     label: 'Inference',     description: 'LLM provider (OpenAI, vLLM, Ollama, Bedrock, etc.)' },
  executor:      { id: 'executor',      label: 'Executor',      description: 'Tool execution dispatcher — routes tool calls to the right backend' },
  vectorio:      { id: 'vectorio',      label: 'VectorIO',      description: 'Semantic search over vector stores — returns ranked chunks with citations' },
  mcp:           { id: 'mcp',           label: 'MCP',           description: 'Model Context Protocol server connection — lazy discovery, session caching' },
  toolruntime:   { id: 'toolruntime',   label: 'ToolRuntime',   description: 'Server-side runtime tool execution (for runtime tools, not function tools)' },
  conversations: { id: 'conversations', label: 'Conversations', description: 'Conversation turn management — stores structured items for UI/metadata' },
  store:         { id: 'store',         label: 'Store',         description: 'SQL persistence for responses and raw chat messages (dual storage)' },
};

/** Fixed ordering so actors always appear in a consistent left-to-right sequence. */
export const ACTOR_ORDER = [
  'client', 'fastapi', 'impl', 'bgworker', 'orchestrator', 'safety',
  'inference', 'executor', 'vectorio', 'mcp', 'toolruntime',
  'conversations', 'store',
];

// ---------------------------------------------------------------------------
// Flow steps
// ---------------------------------------------------------------------------
// Each step is a message arrow in the sequence diagram.
//   from / to  — actor ids
//   label      — arrow label
//   when       — (toggles) => boolean, controls visibility
//   style      — 'request' | 'response' | 'event' | 'async'
//   inLoop     — true if inside the inference loop box

export const FLOW_STEPS = [
  // --- Phase 1: Request arrival ---
  { from: 'client',  to: 'fastapi', label: 'POST /v1/responses',         when: () => true, style: 'request' },
  { from: 'fastapi', to: 'impl',    label: 'create_openai_response()',    when: () => true, style: 'request' },

  // --- Phase 2: Context loading ---
  { from: 'impl',  to: 'store', label: 'get_response_object(prev_id)',    when: t => t.previous_response_id, style: 'request' },
  { from: 'store', to: 'impl',  label: 'previous response + messages',    when: t => t.previous_response_id, style: 'response' },

  { from: 'impl',          to: 'conversations', label: 'list_items(conversation_id)', when: t => t.conversation, style: 'request' },
  { from: 'conversations', to: 'impl',          label: 'conversation items',          when: t => t.conversation, style: 'response' },
  { from: 'impl',  to: 'store', label: 'get_conversation_messages()',     when: t => t.conversation, style: 'request' },
  { from: 'store', to: 'impl',  label: 'stored chat messages',           when: t => t.conversation, style: 'response' },

  // --- Phase 3: Background queueing ---
  { from: 'impl', to: 'store',    label: 'store queued response',     when: t => t.background, style: 'request' },
  { from: 'impl', to: 'bgworker', label: 'enqueue work item',         when: t => t.background, style: 'request' },
  { from: 'impl', to: 'client',   label: 'queued response (immediate)', when: t => t.background, style: 'response' },
  { from: 'bgworker', to: 'orchestrator', label: 'start processing',  when: t => t.background, style: 'async' },

  // --- Non-background: direct orchestration ---
  { from: 'impl', to: 'orchestrator', label: 'create_response()', when: t => !t.background, style: 'request' },

  // --- Phase 4: Streaming event ---
  { from: 'orchestrator', to: 'client', label: 'SSE: response.created', when: t => t.stream && !t.background, style: 'event' },

  // --- Phase 5: Safety — input ---
  { from: 'orchestrator', to: 'safety', label: 'run_guardrails(input)', when: t => t.guardrails, style: 'request' },
  { from: 'safety', to: 'orchestrator', label: 'pass / blocked',        when: t => t.guardrails, style: 'response' },

  // --- Phase 6: MCP tool discovery ---
  { from: 'orchestrator', to: 'mcp', label: 'list_mcp_tools(endpoint)', when: t => t.mcp, style: 'request' },
  { from: 'mcp', to: 'orchestrator', label: 'tool definitions (cached)', when: t => t.mcp, style: 'response' },

  // --- Phase 7: Inference loop ---
  { from: 'orchestrator', to: 'inference',    label: 'openai_chat_completion()',       when: () => true, style: 'request',  inLoop: true },
  { from: 'inference',    to: 'orchestrator',  label: 'completion + tool_calls',       when: () => true, style: 'response', inLoop: true },

  // file_search
  { from: 'orchestrator', to: 'executor', label: 'execute(file_search)',            when: t => t.file_search, style: 'request',  inLoop: true },
  { from: 'executor',     to: 'vectorio', label: 'search_vector_store()',           when: t => t.file_search, style: 'request',  inLoop: true },
  { from: 'vectorio',     to: 'executor', label: 'ranked chunks + citations',       when: t => t.file_search, style: 'response', inLoop: true },
  { from: 'executor',     to: 'orchestrator', label: 'file_search result',          when: t => t.file_search, style: 'response', inLoop: true },

  // MCP tool execution
  { from: 'orchestrator', to: 'executor', label: 'execute(mcp_tool)',               when: t => t.mcp, style: 'request',  inLoop: true },
  { from: 'executor',     to: 'mcp',      label: 'invoke_mcp_tool()',               when: t => t.mcp, style: 'request',  inLoop: true },
  { from: 'mcp',          to: 'executor', label: 'tool result',                     when: t => t.mcp, style: 'response', inLoop: true },
  { from: 'executor',     to: 'orchestrator', label: 'mcp tool result',             when: t => t.mcp, style: 'response', inLoop: true },

  // function tools
  { from: 'orchestrator', to: 'impl', label: 'emit function_call output (breaks loop)', when: t => t.function_tools, style: 'response', inLoop: true },

  // --- Phase 8: Safety — output ---
  { from: 'orchestrator', to: 'safety', label: 'run_guardrails(output)', when: t => t.guardrails, style: 'request' },
  { from: 'safety', to: 'orchestrator', label: 'pass / blocked',         when: t => t.guardrails, style: 'response' },

  // --- Phase 9: Persistence ---
  { from: 'orchestrator', to: 'store', label: 'upsert_response_object()', when: () => true, style: 'request' },

  // --- Phase 10: Conversation sync (dual write) ---
  { from: 'impl', to: 'conversations', label: 'add_items(input + output)',      when: t => t.conversation, style: 'request' },
  { from: 'impl', to: 'store',         label: 'store_conversation_messages()',   when: t => t.conversation, style: 'request' },

  // --- Phase 11: Response delivery ---
  { from: 'orchestrator', to: 'client', label: 'SSE: response.completed | response.incomplete | response.failed', when: t => t.stream && !t.background, style: 'event' },
  { from: 'impl',    to: 'client',  label: 'OpenAIResponseObject',              when: t => !t.stream && !t.background, style: 'response' },

  // Background polling
  { from: 'client',  to: 'fastapi', label: 'GET /v1/responses/{id} (poll)',     when: t => t.background, style: 'async' },
  { from: 'fastapi', to: 'store',   label: 'get_response_object()',             when: t => t.background, style: 'request' },
  { from: 'store',   to: 'fastapi', label: 'completed response',               when: t => t.background, style: 'response' },
  { from: 'fastapi', to: 'client',  label: 'OpenAIResponseObject',             when: t => t.background, style: 'response' },
];

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

export const PRESETS = [
  {
    label: 'Simple completion',
    toggles: {},
  },
  {
    label: 'RAG + conversation',
    toggles: { file_search: true, conversation: true },
  },
  {
    label: 'MCP agent loop',
    toggles: { mcp: true, stream: true },
  },
  {
    label: 'Background processing',
    toggles: { background: true, file_search: true, mcp: true },
  },
  {
    label: 'Full advanced',
    toggles: { stream: true, conversation: true, file_search: true, mcp: true, function_tools: true, guardrails: true },
  },
];

// ---------------------------------------------------------------------------
// Validation rules
// ---------------------------------------------------------------------------

export const VALIDATIONS = [
  {
    check: t => t.previous_response_id && t.conversation,
    message: 'InvalidParameterError: previous_response_id and conversation are both provided — they are mutually exclusive. Use one or the other to provide conversational context.',
    code: 400,
  },
  {
    check: t => t.stream && t.background,
    message: "ValueError: OGX does not yet support 'stream' and 'background' together.",
    code: 400,
  },
];

// ---------------------------------------------------------------------------
// Prose fragments
// ---------------------------------------------------------------------------

const PROSE = {
  base: 'Every request enters through FastAPI routes and is delegated to the Responses provider. The streaming orchestrator manages the inference loop — calling the LLM and executing requested server-side tools until the model produces a final response, emits a client-side function_call, or reaches max_infer_iters.',

  stream: 'With streaming enabled, Server-Sent Events (SSE) are emitted throughout execution. The client receives response.created when processing begins, intermediate events for each tool call and output item, and one terminal event: response.completed, response.incomplete, or response.failed.',

  background: 'In background mode, the request is immediately queued and a response with status "queued" is returned to the client. One of 10 async workers picks up the job, runs the full inference loop internally, and updates the stored response. The client polls GET /v1/responses/{id} to check progress and retrieve the completed response.',

  previous_response_id: 'The previous response is loaded from the Store, including its original input and the full chat message history. These are prepended to the current input, giving the model conversational context without requiring the client to resend the entire history.',

  conversation: 'Conversation state is loaded from two sources: the Conversations API provides structured turn items (for UI and metadata), while the Store provides the raw chat messages used for inference. After the response completes, both stores are updated — items are added to the conversation, and the full message array is persisted for the next turn. This dual-storage pattern enables both conversation-level UI and accurate inference continuity.',

  file_search: 'When the model requests a file search, the Executor queries VectorIO\'s search_vector_store() endpoint with the model\'s query. VectorIO searches the configured vector stores and returns ranked document chunks with relevance scores. These are formatted with citations and fed back to the model as tool results for the next inference iteration.',

  mcp: 'MCP (Model Context Protocol) tools are discovered lazily — the orchestrator calls list_mcp_tools() on the configured server endpoint when MCP tools first appear. Tool definitions are cached for the duration of the request via MCPSessionManager. When the model invokes an MCP tool, the executor reuses the existing session, avoiding redundant connection setup.',

  function_tools: 'Function tools are client-side in the Responses flow. When the model emits a function tool call, OGX returns it as a function_call output item and exits the inference loop. Your client executes the function and sends the next request with a function_call_output item to continue.',

  guardrails: 'Safety guardrails validate both input and output. Before the inference loop begins, the combined input text is checked against configured guardrail policies. If a violation is detected, the response is refused immediately. After the model generates output, the response text is checked again — an output violation is converted into a refusal response with violation details.',
};

export function getProse(toggles) {
  const parts = [PROSE.base];
  if (toggles.previous_response_id) parts.push(PROSE.previous_response_id);
  if (toggles.conversation)         parts.push(PROSE.conversation);
  if (toggles.background)           parts.push(PROSE.background);
  if (toggles.stream)               parts.push(PROSE.stream);
  if (toggles.guardrails)           parts.push(PROSE.guardrails);
  if (toggles.file_search)          parts.push(PROSE.file_search);
  if (toggles.mcp)                  parts.push(PROSE.mcp);
  if (toggles.function_tools)       parts.push(PROSE.function_tools);
  return parts;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return the ordered list of actor objects that appear in the given steps. */
export function getActiveActors(steps) {
  const seen = new Set();
  for (const s of steps) {
    seen.add(s.from);
    seen.add(s.to);
  }
  return ACTOR_ORDER.filter(id => seen.has(id)).map(id => ACTORS[id]);
}
