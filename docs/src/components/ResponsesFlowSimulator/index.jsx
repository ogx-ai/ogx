import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import { useColorMode } from '@docusaurus/theme-common';
import styles from './styles.module.css';
import {
  ACTORS,
  FLOW_STEPS,
  PRESETS,
  VALIDATIONS,
  getProse,
  getActiveActors,
} from './flowData';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const ACTOR_W   = 106;
const ACTOR_H   = 30;
const ACTOR_GAP = 14;
const MSG_H     = 30;
const PAD       = 20;
const HEAD_GAP  = 14;
const ARROW_PAD = 6;   // padding between lifeline and arrow start/end
const HEAD_LEN  = 7;   // arrowhead size
const HEAD_W    = 4;

const DEFAULT_TOGGLES = {
  stream: false,
  background: false,
  previous_response_id: false,
  conversation: false,
  file_search: false,
  mcp: false,
  function_tools: false,
  guardrails: false,
};

const TOGGLE_LABELS = {
  stream:               'stream',
  background:           'background',
  previous_response_id: 'previous_response_id',
  conversation:         'conversation',
  file_search:          'file_search',
  mcp:                  'mcp',
  function_tools:       'function tools',
  guardrails:           'guardrails',
};

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

/** Horizontal arrow with manual arrowhead (full CSS color control). */
function HArrow({ x1, x2, y, styleType, dashed }) {
  const dir = x2 > x1 ? 1 : -1;
  const tipX = x2;
  const baseX = x2 - dir * HEAD_LEN;

  const dashProps = dashed ? { strokeDasharray: '5 3' } : {};

  return (
    <g className={styles[`arrow_${styleType}`]}>
      <line
        x1={x1}
        y1={y}
        x2={baseX}
        y2={y}
        strokeWidth={1.4}
        stroke="currentColor"
        {...dashProps}
      />
      <polygon
        points={`${tipX},${y} ${baseX},${y - HEAD_W} ${baseX},${y + HEAD_W}`}
        fill="currentColor"
      />
    </g>
  );
}

// ---------------------------------------------------------------------------
// SequenceDiagram
// ---------------------------------------------------------------------------

function SequenceDiagram({ actors, steps, toggleKey, hoveredActor, onHoverActor }) {
  const actorIndex = {};
  actors.forEach((a, i) => { actorIndex[a.id] = i; });

  const actorX = id => PAD + actorIndex[id] * (ACTOR_W + ACTOR_GAP) + ACTOR_W / 2;
  const headerArea = PAD + ACTOR_H + HEAD_GAP;
  const msgY = i => headerArea + i * MSG_H + MSG_H / 2;

  const width  = PAD * 2 + actors.length * ACTOR_W + (actors.length - 1) * ACTOR_GAP;
  const height = headerArea + steps.length * MSG_H + PAD;

  // Loop box bounds
  const loopIndices = steps.reduce((acc, s, i) => { if (s.inLoop) acc.push(i); return acc; }, []);
  const hasLoop = loopIndices.length > 0;
  const loopFirst = hasLoop ? loopIndices[0] : 0;
  const loopLast  = hasLoop ? loopIndices[loopIndices.length - 1] : 0;

  // Loop box geometry
  const loopBoxX = PAD - 10;
  const loopBoxY = msgY(loopFirst) - MSG_H / 2 - 6;
  const loopBoxW = width - PAD * 2 + 20;
  const loopBoxH = (loopLast - loopFirst + 1) * MSG_H + 12;
  const loopLabelH = 16;

  return (
    <svg
      key={toggleKey}
      width={width}
      height={height}
      className={styles.svg}
      role="img"
      aria-label="Responses API sequence diagram"
    >
      {/* Actor headers */}
      {actors.map((actor) => {
        const cx = actorX(actor.id);
        const isHovered = hoveredActor === actor.id;
        return (
          <g
            key={actor.id}
            onMouseEnter={() => onHoverActor(actor.id)}
            onMouseLeave={() => onHoverActor(null)}
            style={{ cursor: 'default' }}
          >
            <rect
              x={cx - ACTOR_W / 2}
              y={PAD}
              width={ACTOR_W}
              height={ACTOR_H}
              rx={6}
              className={`${styles.actorBox} ${isHovered ? styles.actorBoxHover : ''}`}
            />
            <text
              x={cx}
              y={PAD + ACTOR_H / 2 + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              className={styles.actorLabel}
            >
              {actor.label}
            </text>
          </g>
        );
      })}

      {/* Lifelines */}
      {actors.map((actor) => {
        const cx = actorX(actor.id);
        return (
          <line
            key={`ll-${actor.id}`}
            x1={cx}
            y1={PAD + ACTOR_H}
            x2={cx}
            y2={height - PAD + 8}
            className={styles.lifeline}
          />
        );
      })}

      {/* Loop box */}
      {hasLoop && (
        <g
          className={styles.loopBox}
          style={{ '--loop-delay': `${loopFirst * 80 + 40}ms` }}
        >
          <rect
            x={loopBoxX}
            y={loopBoxY}
            width={loopBoxW}
            height={loopBoxH}
            rx={4}
          />
          <rect
            x={loopBoxX}
            y={loopBoxY}
            width={188}
            height={loopLabelH}
            rx={3}
            className={styles.loopLabelBg}
          />
          <text
            x={loopBoxX + 6}
            y={loopBoxY + loopLabelH / 2 + 1}
            dominantBaseline="middle"
            className={styles.loopLabel}
          >
            loop [until no tool_calls or max_iters]
          </text>
        </g>
      )}

      {/* Messages */}
      {steps.map((step, i) => {
        const fromX = actorX(step.from);
        const toX   = actorX(step.to);
        const y     = msgY(i);
        const dashed = step.style === 'event' || step.style === 'async';

        // Label position — centered above arrow
        const labelX = (fromX + toX) / 2;
        const labelY = y - 7;

        return (
          <g
            key={`msg-${i}`}
            className={styles.messageRow}
            style={{ '--delay': `${i * 80}ms` }}
          >
            <HArrow
              x1={fromX + (toX > fromX ? ARROW_PAD : -ARROW_PAD)}
              x2={toX   + (toX > fromX ? -ARROW_PAD : ARROW_PAD)}
              y={y}
              styleType={step.style}
              dashed={dashed}
            />
            <text
              x={labelX}
              y={labelY}
              textAnchor="middle"
              className={styles.messageLabel}
            >
              {step.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// TogglePanel
// ---------------------------------------------------------------------------

function TogglePanel({ toggles, onChange, onPreset, activePreset }) {
  return (
    <div className={styles.togglePanel}>
      <div className={styles.presetRow}>
        {PRESETS.map((p, i) => (
          <button
            key={p.label}
            type="button"
            className={`${styles.preset} ${activePreset === i ? styles.presetActive : ''}`}
            onClick={() => onPreset(i)}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className={styles.toggleRow}>
        {Object.keys(TOGGLE_LABELS).map(key => (
          <label key={key} className={styles.toggle}>
            <input
              type="checkbox"
              checked={toggles[key]}
              onChange={() => onChange(key)}
            />
            <span className={styles.toggleCheck} />
            <code className={styles.toggleCode}>{TOGGLE_LABELS[key]}</code>
          </label>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProsePanel
// ---------------------------------------------------------------------------

function ProsePanel({ paragraphs }) {
  return (
    <div className={styles.prosePanel}>
      {paragraphs.map((p, i) => (
        <p key={i} className={styles.proseParagraph}>{p}</p>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

function ValidationError({ rule }) {
  return (
    <div className={styles.errorContainer}>
      <div className={styles.errorHeader}>
        <span className={styles.errorCode}>{rule.code}</span>
        <span className={styles.errorTitle}>Validation Error</span>
      </div>
      <code className={styles.errorMessage}>{rule.message}</code>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ActorTooltip
// ---------------------------------------------------------------------------

function ActorTooltip({ actor, actors }) {
  if (!actor) return null;
  const idx = actors.findIndex(a => a.id === actor.id);
  if (idx < 0) return null;

  // Compute position from the same constants used by the SVG layout,
  // offset by the diagram scroll container padding (0.75rem ≈ 12px).
  const scrollPad = 12;
  const cx = scrollPad + PAD + idx * (ACTOR_W + ACTOR_GAP) + ACTOR_W / 2;
  const ty = scrollPad + PAD + ACTOR_H + 6;

  return (
    <div
      className={styles.tooltip}
      style={{ left: cx, top: ty }}
    >
      <strong>{actor.label}</strong>
      <span>{actor.description}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CodeExample
// ---------------------------------------------------------------------------

function generateCode(toggles) {
  const lines = [
    'from openai import OpenAI',
    '',
    'client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")',
  ];

  // Build the tools array
  const tools = [];
  if (toggles.file_search) {
    tools.push(`        {
            "type": "file_search",
            "vector_store_ids": ["vs_abc123"],
        }`);
  }
  if (toggles.mcp) {
    tools.push(`        {
            "type": "mcp",
            "server_label": "github",
            "server_url": "http://localhost:8080/sse",
        }`);
  }
  if (toggles.function_tools) {
    tools.push(`        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }`);
  }

  // Build kwargs
  const kwargs = [];
  kwargs.push('    model="llama-3.3-70b",');

  if (toggles.previous_response_id) {
    kwargs.push('    previous_response_id="resp_abc123",');
    kwargs.push('    input="Now summarize the key findings",');
  } else if (toggles.file_search) {
    kwargs.push('    input="What do the uploaded documents say about Q4 results?",');
  } else if (toggles.mcp) {
    kwargs.push('    input="List open issues in the repository",');
  } else if (toggles.function_tools) {
    kwargs.push('    input="What is the weather in Paris?",');
  } else {
    kwargs.push('    input="Explain how transformers work",');
  }

  if (tools.length > 0) {
    kwargs.push('    tools=[');
    kwargs.push(tools.join(',\n'));
    kwargs.push('    ],');
  }

  if (toggles.conversation) {
    kwargs.push('    conversation="my-session-001",');
  }
  if (toggles.stream) {
    kwargs.push('    stream=True,');
  }
  if (toggles.background) {
    kwargs.push('    background=True,');
  }
  if (toggles.guardrails) {
    kwargs.push('    guardrail_ids=["content-safety"],');
  }

  lines.push('response = client.responses.create(');
  lines.push(...kwargs);
  lines.push(')');

  // Add usage pattern based on mode
  if (toggles.stream && !toggles.background) {
    lines.push('');
    lines.push('for event in response:');
    lines.push('    print(event)');
  } else if (toggles.background) {
    lines.push('');
    lines.push('# Poll until complete');
    lines.push('import time');
    lines.push('while response.status in ("queued", "in_progress"):');
    lines.push('    time.sleep(1)');
    lines.push('    response = client.responses.retrieve(response.id)');
    lines.push('');
    lines.push('print(response.output)');
  } else if (toggles.function_tools) {
    lines.push('');
    lines.push('# Handle function call output');
    lines.push('for item in response.output:');
    lines.push('    if item.type == "function_call":');
    lines.push('        result = call_my_function(item.name, item.arguments)');
    lines.push('        response = client.responses.create(');
    lines.push('            model="llama-3.3-70b",');
    lines.push('            previous_response_id=response.id,');
    lines.push('            input=[{');
    lines.push('                "type": "function_call_output",');
    lines.push('                "call_id": item.call_id,');
    lines.push('                "output": result,');
    lines.push('            }],');
    lines.push('        )');
  } else {
    lines.push('');
    lines.push('print(response.output_text)');
  }

  return lines.join('\n');
}

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="10" height="10" rx="2" />
      <path d="M5 15V7a2 2 0 0 1 2-2h8" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
      <path d="m5 13 4 4L19 7" />
    </svg>
  );
}

function CodeExample({ toggles, toggleKey }) {
  const code = useMemo(() => generateCode(toggles), [toggles]);
  const [copied, setCopied] = useState(false);
  const { colorMode } = useColorMode();
  const isDark = colorMode === 'dark';

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(id);
  }, [copied]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
    } catch { /* fallback */ }
  };

  return (
    <div className={`${styles.codeContainer} ${isDark ? styles.codeContainerDark : styles.codeContainerLight}`}>
      <div className={styles.codeHeader}>
        <span className={styles.codeTitle}>Python</span>
        <button
          type="button"
          className={`${styles.codeCopy} ${copied ? styles.codeCopyCopied : ''}`}
          onClick={handleCopy}
          aria-label="Copy code"
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <Highlight theme={isDark ? themes.oneDark : themes.oneLight} code={code} language="python" key={`${toggleKey}-${colorMode}`}>
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className={styles.codePre}>
            <code className={styles.codeBlock}>
              {tokens.map((line, i) => (
                <span key={i} {...getLineProps({ line })}>
                  {line.map((token, j) => (
                    <span key={j} {...getTokenProps({ token })} />
                  ))}
                  {'\n'}
                </span>
              ))}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ResponsesFlowSimulator() {
  const [toggles, setToggles]           = useState(DEFAULT_TOGGLES);
  const [activePreset, setActivePreset] = useState(0);
  const [hoveredActorId, setHoveredActorId] = useState(null);
  // Validation
  const validationError = useMemo(() => {
    for (const rule of VALIDATIONS) {
      if (rule.check(toggles)) return rule;
    }
    return null;
  }, [toggles]);

  // Visible steps & actors
  const visibleSteps = useMemo(() => {
    if (validationError) return [];
    return FLOW_STEPS.filter(s => s.when(toggles));
  }, [toggles, validationError]);

  const activeActors = useMemo(() => getActiveActors(visibleSteps), [visibleSteps]);

  // Prose
  const prose = useMemo(() => getProse(toggles), [toggles]);

  // Animation key
  const toggleKey = useMemo(() => JSON.stringify(toggles), [toggles]);

  // Hovered actor object
  const hoveredActor = hoveredActorId ? ACTORS[hoveredActorId] : null;

  // Handlers
  const handleToggle = useCallback((key) => {
    setToggles(prev => ({ ...prev, [key]: !prev[key] }));
    setActivePreset(null);
  }, []);

  const handlePreset = useCallback((index) => {
    setToggles({ ...DEFAULT_TOGGLES, ...PRESETS[index].toggles });
    setActivePreset(index);
  }, []);

  return (
    <div className={styles.container}>
      <TogglePanel
        toggles={toggles}
        onChange={handleToggle}
        onPreset={handlePreset}
        activePreset={activePreset}
      />

      {validationError ? (
        <ValidationError rule={validationError} />
      ) : (
        <>
          <div className={styles.diagramContainer}>
            <div className={styles.diagramScroll}>
              <SequenceDiagram
                actors={activeActors}
                steps={visibleSteps}
                toggleKey={toggleKey}
                hoveredActor={hoveredActorId}
                onHoverActor={setHoveredActorId}
              />
            </div>
            {hoveredActor && (
              <ActorTooltip
                actor={hoveredActor}
                actors={activeActors}
              />
            )}
          </div>
          <CodeExample toggles={toggles} toggleKey={toggleKey} />
          <ProsePanel paragraphs={prose} />
        </>
      )}
    </div>
  );
}
