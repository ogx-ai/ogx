import React, {useState, useEffect, useRef} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import {useColorMode} from '@docusaurus/theme-common';
import {Highlight, themes} from 'prism-react-renderer';
import InstallBlock from '../components/InstallBlock';
import styles from './index.module.css';

const LANG_TO_PRISM = {
  'Python': 'python',
  'curl': 'bash',
  'Node.js': 'javascript',
  'Go': 'go',
  'TypeScript': 'typescript',
};

const SDK_EXAMPLES = {
  openai: {
    label: 'OpenAI SDK',
    endpoint: '/v1/responses',
    languages: [
      {
        lang: 'Python',
        code: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")
response = client.responses.create(
    model="llama-3.3-70b",
    input="Summarize this repository",
    tools=[{"type": "web_search"}],
)`,
      },
      {
        lang: 'curl',
        code: `curl http://localhost:8321/v1/responses \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-3.3-70b",
    "input": "Summarize this repository",
    "tools": [{"type": "web_search"}]
  }'`,
      },
      {
        lang: 'Node.js',
        code: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8321/v1",
  apiKey: "fake",
});

const response = await client.responses.create({
  model: "llama-3.3-70b",
  input: "Summarize this repository",
  tools: [{ type: "web_search" }],
});`,
      },
      {
        lang: 'Go',
        code: `client := openai.NewClient(
  option.WithBaseURL("http://localhost:8321/v1"),
  option.WithAPIKey("fake"),
)

response, err := client.Responses.New(
  context.TODO(),
  openai.ResponseNewParams{
    Model: "llama-3.3-70b",
    Input: "Summarize this repository",
    Tools: []ResponseToolUnionParam{
      openai.WebSearchTool(),
    },
  },
)`,
      },
    ],
  },
  anthropic: {
    label: 'Anthropic SDK',
    endpoint: '/v1/messages',
    languages: [
      {
        lang: 'Python',
        code: `from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8321/v1",
    api_key="fake",
)
message = client.messages.create(
    model="llama-3.3-70b",
    max_tokens=1024,
    messages=[
        {"role": "user",
         "content": "Summarize this repository"}
    ],
)`,
      },
      {
        lang: 'TypeScript',
        code: `import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  baseURL: "http://localhost:8321/v1",
  apiKey: "fake",
});

const message = await client.messages.create({
  model: "llama-3.3-70b",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Summarize this repository" }],
});`,
      },
    ],
  },
  google: {
    label: 'Google GenAI',
    endpoint: '/v1alpha/interactions',
    languages: [
      {
        lang: 'Python',
        code: `from google import genai
from google.genai import types

client = genai.Client(
    api_key="fake",
    http_options=types.HttpOptions(
        base_url="http://localhost:8321",
        api_version="v1alpha",
    ),
)
interaction = client.interactions.create(
    model="llama-3.3-70b",
    input="Summarize this repository",
)`,
      },
    ],
  },
};

const API_SURFACE = [
  { category: 'Inference', endpoints: [
    { label: 'Chat Completions', path: '/v1/chat/completions' },
    { label: 'Responses', path: '/v1/responses' },
    { label: 'Embeddings', path: '/v1/embeddings' },
    { label: 'Models', path: '/v1/models' },
    { label: 'Messages', path: '/v1/messages', note: 'Anthropic' },
    { label: 'Interactions', path: '/v1alpha/interactions', note: 'Google' },
  ]},
  { category: 'Data', endpoints: [
    { label: 'Vector Stores', path: '/v1/vector_stores' },
    { label: 'Files', path: '/v1/files' },
    { label: 'Batches', path: '/v1/batches' },
  ]},
  { category: 'Moderation & Tools', endpoints: [
    { label: 'Moderations', path: '/v1/moderations' },
    { label: 'Tools', path: '/v1/tools' },
    { label: 'Connectors', path: '/v1/connectors' },
  ]},
];

const PROVIDERS = [
  { name: 'Ollama', href: '/docs/providers/inference/remote_ollama' },
  { name: 'vLLM', href: '/docs/providers/inference/remote_llama-openai-compat' },
  { name: 'OpenAI', href: '/docs/providers/inference/remote_openai' },
  { name: 'Anthropic', href: '/docs/providers/inference/remote_anthropic' },
  { name: 'AWS Bedrock', href: '/docs/providers/inference/remote_bedrock' },
  { name: 'Azure OpenAI', href: '/docs/providers/inference/remote_azure' },
  { name: 'Gemini', href: '/docs/providers/inference/remote_gemini' },
  { name: 'Together AI', href: '/docs/providers/inference/remote_together' },
  { name: 'Fireworks', href: '/docs/providers/inference/remote_fireworks' },
  { name: 'PGVector', href: '/docs/providers/vector_io/remote_pgvector' },
  { name: 'Qdrant', href: '/docs/providers/vector_io/remote_qdrant' },
  { name: 'ChromaDB', href: '/docs/providers/vector_io/remote_chromadb' },
  { name: 'Milvus', href: '/docs/providers/vector_io/remote_milvus' },
  { name: 'Weaviate', href: '/docs/providers/vector_io/remote_weaviate' },
];

const CLI_DEMOS = {
  claude: {
    label: 'Claude Code',
    command: 'claude',
    envVar: 'ANTHROPIC_BASE_URL',
    lines: [
      { type: 'command', text: '$ export ANTHROPIC_BASE_URL=http://localhost:8321/v1', delay: 0 },
      { type: 'command', text: '$ claude', delay: 400 },
      { type: 'blank', text: '', delay: 100 },
      { type: 'brand', text: ' ▐▛███▜▌   Claude Code', delay: 0 },
      { type: 'brand-dim', text: '▝▜█████▛▘  llama-3.3-70b via OGX', delay: 0 },
      { type: 'brand-dim', text: '  ▘▘ ▝▝    ~/my-project', delay: 0 },
      { type: 'blank', text: '', delay: 300 },
      { type: 'prompt', text: '> hey big dawg', delay: 300 },
      { type: 'blank', text: '', delay: 400 },
      { type: 'result', text: 'Hello from OGX!', delay: 0 },
    ],
  },
  codex: {
    label: 'Codex',
    command: 'codex',
    envVar: 'OPENAI_BASE_URL',
    lines: [
      { type: 'command', text: '$ cat ~/.codex/config.toml', delay: 0 },
      { type: 'brand-dim', text: '[model_providers.ogx]', delay: 0 },
      { type: 'brand-dim', text: 'base_url = "http://localhost:8321/v1"', delay: 0 },
      { type: 'blank', text: '', delay: 200 },
      { type: 'command', text: '$ codex', delay: 400 },
      { type: 'blank', text: '', delay: 100 },
      { type: 'brand-box', text: '>_ OpenAI Codex\n\nmodel: llama-3.3-70b via OGX', delay: 0 },
      { type: 'blank', text: '', delay: 300 },
      { type: 'prompt', text: '> hey big dawg', delay: 300 },
      { type: 'blank', text: '', delay: 400 },
      { type: 'result', text: 'Hello from OGX!', delay: 0 },
    ],
  },
};

/* Custom Tidal theme for the landing page code block */
const tidalDark = {
  plain: { color: '#bcc5d0', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog'], style: { color: '#546678' } },
    { types: ['keyword', 'builtin'], style: { color: '#7eb8d4' } },
    { types: ['string', 'attr-value', 'char'], style: { color: '#d4a55a' } },
    { types: ['function'], style: { color: '#2dbdc2' } },
    { types: ['class-name'], style: { color: '#c9a84c' } },
    { types: ['number', 'boolean'], style: { color: '#c9a84c' } },
    { types: ['operator'], style: { color: '#45cace' } },
    { types: ['punctuation'], style: { color: '#8b98a8' } },
    { types: ['property', 'constant'], style: { color: '#d4856a' } },
    { types: ['variable'], style: { color: '#bcc5d0' } },
  ],
};

const tidalLight = {
  plain: { color: '#2d3748', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog'], style: { color: '#8393a7' } },
    { types: ['keyword', 'builtin'], style: { color: '#3d6b8e' } },
    { types: ['string', 'attr-value', 'char'], style: { color: '#8a6e2f' } },
    { types: ['function'], style: { color: '#0b6165' } },
    { types: ['class-name'], style: { color: '#7c6322' } },
    { types: ['number', 'boolean'], style: { color: '#7c6322' } },
    { types: ['operator'], style: { color: '#0d7377' } },
    { types: ['punctuation'], style: { color: '#4a5568' } },
    { types: ['property', 'constant'], style: { color: '#8b4e3a' } },
    { types: ['variable'], style: { color: '#2d3748' } },
  ],
};

function useScrollReveal() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mq.matches) { setVisible(true); return; }
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect(); } },
      { threshold: 0.15 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return { ref, className: visible ? styles.revealed : styles.reveal };
}

function Section({children, className, ...props}) {
  const scroll = useScrollReveal();
  return (
    <section ref={scroll.ref} className={clsx(scroll.className, className)} {...props}>
      {children}
    </section>
  );
}

function CodeCopyButton({text}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(id);
  }, [copied]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      /* fallback */
    }
  };

  return (
    <button
      type="button"
      className={clsx(styles.codeCopyBtn, copied && styles.codeCopyBtnCopied)}
      onClick={handleCopy}
      aria-label="Copy code"
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
          <path d="m5 13 4 4L19 7" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="10" height="10" rx="2" />
          <path d="M5 15V7a2 2 0 0 1 2-2h8" />
        </svg>
      )}
      <span>{copied ? 'Copied' : 'Copy'}</span>
    </button>
  );
}

function CodeBlock() {
  const [activeSdk, setActiveSdk] = useState('openai');
  const [langIndex, setLangIndex] = useState({openai: 0, anthropic: 0, google: 0});
  const {colorMode} = useColorMode();
  const sdk = SDK_EXAMPLES[activeSdk];
  const activeIdx = langIndex[activeSdk];
  const prismLang = LANG_TO_PRISM[sdk.languages[activeIdx].lang] || 'bash';
  const theme = colorMode === 'dark' ? tidalDark : tidalLight;

  return (
    <div className={styles.codeBlock}>
      <div className={styles.codeHeader}>
        <div className={styles.sdkTabs}>
          {Object.entries(SDK_EXAMPLES).map(([key, val]) => (
            <button
              key={key}
              className={clsx(styles.sdkTab, activeSdk === key && styles.sdkTabActive)}
              onClick={() => setActiveSdk(key)}
            >
              {val.label}
            </button>
          ))}
        </div>
        <div className={styles.codeHeaderRight}>
          <CodeCopyButton text={sdk.languages[activeIdx].code} />
          <code className={styles.endpointLabel}>{sdk.endpoint}</code>
        </div>
      </div>
      <div className={styles.langTabs}>
        {sdk.languages.map((ex, i) => (
          <button
            key={ex.lang}
            className={clsx(styles.langTab, i === activeIdx && styles.langTabActive)}
            onClick={() => setLangIndex(prev => ({...prev, [activeSdk]: i}))}
          >
            {ex.lang}
          </button>
        ))}
      </div>
      <div className={styles.codeFade} key={`${activeSdk}-${activeIdx}`}>
        <Highlight theme={theme} code={sdk.languages[activeIdx].code} language={prismLang}>
          {({style, tokens, getLineProps, getTokenProps}) => (
            <pre className={styles.codeContent} style={style}>
              <code>
                {tokens.map((line, i) => (
                  <div key={i} {...getLineProps({line})}>
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({token})} />
                    ))}
                  </div>
                ))}
              </code>
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}

function useTerminalAnimation(lines, shouldStart) {
  const [visibleLines, setVisibleLines] = useState([]);
  const [typingLine, setTypingLine] = useState(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!shouldStart) return;
    let cancelled = false;

    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mq.matches) {
      setVisibleLines(lines.map(l => l.text));
      setTypingLine(null);
      setDone(true);
      return;
    }

    async function animate() {
      const shown = [];
      for (let i = 0; i < lines.length; i++) {
        if (cancelled) return;
        const line = lines[i];

        if (line.type === 'blank') {
          shown.push('');
          setVisibleLines([...shown]);
          await sleep(line.delay);
          continue;
        }

        if (line.type === 'command' || line.type === 'prompt') {
          let partial = '';
          for (let c = 0; c < line.text.length; c++) {
            if (cancelled) return;
            partial += line.text[c];
            setTypingLine(partial);
            await sleep(30 + Math.random() * 20);
          }
          setTypingLine(null);
          shown.push(line.text);
          setVisibleLines([...shown]);
          await sleep(line.delay);
        } else {
          await sleep(line.delay);
          if (cancelled) return;
          shown.push(line.text);
          setVisibleLines([...shown]);
        }
      }
      setDone(true);
    }

    animate();
    return () => { cancelled = true; };
  }, [shouldStart, lines]);

  return { visibleLines, typingLine, done };
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function CliShowcase() {
  const [started, setStarted] = useState(false);
  const sectionRef = useRef(null);

  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mq.matches) { setStarted(true); return; }
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setStarted(true); observer.disconnect(); } },
      { threshold: 0.15 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const claudeAnim = useTerminalAnimation(CLI_DEMOS.claude.lines, started);
  const codexAnim = useTerminalAnimation(CLI_DEMOS.codex.lines, claudeAnim.done);

  return (
    <section className={styles.cliSection} ref={sectionRef}>
      <div className="container">
        <div className={styles.cliHeader}>
          <h2>Your tools. Any model.</h2>
          <p>
            Configure OGX with any provider — Ollama, vLLM, Bedrock, Azure, or
            your own. Then point{' '}
            <a href="https://ogx-ai.github.io/docs/building_applications/claude_code_integration">Claude Code</a>,{' '}
            <a href="https://ogx-ai.github.io/docs/building_applications/codex_cli_integration">Codex</a>, or{' '}
            <a href="https://ogx-ai.github.io/blog/opencode-blog">OpenCode</a>{' '}
            at it. Same workflow, any model.
          </p>
        </div>
        <div className={styles.cliTerminals}>
          <TerminalWindow demo={CLI_DEMOS.claude} anim={claudeAnim} />
          <TerminalWindow demo={CLI_DEMOS.codex} anim={codexAnim} />
        </div>
      </div>
    </section>
  );
}

function TerminalWindow({demo, anim}) {
  return (
    <div className={clsx(styles.cliTerminal, styles.cliTerminalFadeIn)}>
      <div className={styles.cliTerminalBar}>
        <span className={styles.cliTerminalFlow}>
          {demo.label} → OGX
        </span>
      </div>
      <div className={styles.cliTerminalBody}>
        {anim.visibleLines.map((line, i) => {
          const meta = demo.lines[i];
          if (!meta) return null;
          const cls = {
            command: styles.cliLineCommand,
            prompt: styles.cliLinePrompt,
            status: styles.cliLineStatus,
            result: styles.cliLineResult,
            'result-cont': styles.cliLineResultCont,
            blank: styles.cliLineBlank,
            brand: styles.cliLineBrand,
            'brand-dim': styles.cliLineBrandDim,
            'brand-box': styles.cliLineBrandBox,
          }[meta.type] || '';
          if (meta.type === 'blank') return <div key={i} className={styles.cliLineBlank} />;
          if (meta.type === 'brand-box') {
            return (
              <div key={i} className={styles.cliLineBrandBox}>
                {line.split('\n').map((l, j) => (
                  <div key={j}>{l || ' '}</div>
                ))}
              </div>
            );
          }
          return (
            <div key={i} className={clsx(styles.cliLine, cls)}>{line}</div>
          );
        })}
        {anim.typingLine !== null && (
          <div className={clsx(styles.cliLine, styles.cliLineCommand)}>
            {anim.typingLine}
            <span className={styles.cliCursor} />
          </div>
        )}
      </div>
    </div>
  );
}

function useConstellation(canvasId) {
  useEffect(() => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mq.matches) return;

    const ctx = canvas.getContext('2d');
    let raf;
    let nodes = [];
    const NODE_COUNT = 40;
    const CONNECT_DIST = 140;
    const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark';

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    function initNodes() {
      const rect = canvas.parentElement.getBoundingClientRect();
      nodes = [];
      for (let i = 0; i < NODE_COUNT; i++) {
        nodes.push({
          x: Math.random() * rect.width,
          y: Math.random() * rect.height,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          r: 1.5 + Math.random() * 1.5,
        });
      }
    }

    function draw() {
      const rect = canvas.parentElement.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const dark = isDark();
      const dotColor = dark ? 'rgba(45, 189, 194, 0.4)' : 'rgba(13, 115, 119, 0.25)';
      const lineColor = dark ? 'rgba(45, 189, 194,' : 'rgba(13, 115, 119,';

      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        a.x += a.vx;
        a.y += a.vy;
        if (a.x < 0 || a.x > w) a.vx *= -1;
        if (a.y < 0 || a.y > h) a.vy *= -1;

        ctx.beginPath();
        ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
        ctx.fillStyle = dotColor;
        ctx.fill();

        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECT_DIST) {
            const opacity = (1 - dist / CONNECT_DIST) * (dark ? 0.15 : 0.1);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = lineColor + opacity + ')';
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(draw);
    }

    resize();
    initNodes();
    draw();

    const ro = new ResizeObserver(() => {
      resize();
    });
    ro.observe(canvas.parentElement);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [canvasId]);
}

function AnnouncementBanner() {
  const [visible, setVisible] = useState(true);
  const [dismissing, setDismissing] = useState(false);

  if (!visible) return null;

  const handleDismiss = () => {
    setDismissing(true);
    setTimeout(() => setVisible(false), 350);
  };

  return (
    <div className={clsx(styles.announcementBar, dismissing && styles.announcementDismissing)}>
      <div className="container">
        <div className={styles.announcementInner}>
          <span className={styles.announcementPulse} aria-hidden="true" />
          <span className={styles.announcementLabel}>New</span>
          <span className={styles.announcementSep} aria-hidden="true" />
          <span className={styles.announcementText}>
            Llama Stack is now <span className={styles.announcementHighlight}>OGX</span>
          </span>
          <a
            className={styles.announcementLink}
            href="https://ogx-ai.github.io/blog/from-llama-stack-to-ogx"
          >
            Read the story <span className={styles.announcementArrow}>&rarr;</span>
          </a>
          <button
            type="button"
            className={styles.announcementDismiss}
            onClick={handleDismiss}
            aria-label="Dismiss announcement"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18" /><path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function Hero() {
  useConstellation('hero-constellation');

  return (
    <section className={styles.hero}>
      <canvas className={styles.heroCanvas} id="hero-constellation" aria-hidden="true" />
      <div className="container">
        <div className={styles.heroLayout}>
          <div className={styles.heroText}>
            <h1 className={styles.title}>
              Not a gateway.<br />
              The full stack.
            </h1>
            <p className={styles.subtitle}>
              Inference, vector stores, file storage, moderation, tool calling,
              and agentic orchestration — as a server or a Python library.
              Pluggable providers, any language, deploy anywhere.
            </p>
            <InstallBlock />
            <div className={styles.actions}>
              <Link className={styles.primaryBtn} to="/docs/getting_started/quickstart">
                Get started
              </Link>
              <Link className={styles.secondaryBtn} to="/docs/api-openai">
                API docs
              </Link>
              <a className={styles.githubBtn} href="https://github.com/ogx-ai/ogx" target="_blank" rel="noopener noreferrer">
                GitHub
              </a>
            </div>
          </div>
          <div className={styles.heroCode}>
            <CodeBlock />
          </div>
        </div>
      </div>
    </section>
  );
}

function ApiSurface() {
  return (
    <Section className={styles.apiSection}>
      <div className="container">
        <div className={styles.apiHeader}>
          <h2>Everything your AI app needs. One process.</h2>
          <p>
            More than inference routing. OGX composes inference, storage,
            moderation, and orchestration into a single process — whether you
            run it as a server or import it as a library. Your agent can search
            a vector store, call a tool, apply moderation checks, and stream the response.
            No glue code. No sidecar services.
          </p>
        </div>
        <div className={styles.apiColumns}>
          {API_SURFACE.map(group => (
            <div key={group.category} className={styles.apiGroup}>
              <h3 className={styles.apiGroupTitle}>{group.category}</h3>
              <div className={styles.endpointList}>
                {group.endpoints.map(ep => (
                  <div key={ep.path} className={styles.endpointRow}>
                    <code className={styles.endpointPath}>{ep.path}</code>
                    <span className={styles.endpointName}>
                      {ep.label}
                      {ep.note && <span className={styles.endpointNote}>{ep.note}</span>}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <Link className={styles.textLink} to="/docs/api-openai">
          Full API reference
        </Link>
      </div>
    </Section>
  );
}

function ServerAndLibrary() {
  return (
    <Section className={styles.serverSection}>
      <div className="container">
        <div className={styles.serverLayout}>
          <div>
            <h2>Server or library. Your call.</h2>
            <p>
              Deploy OGX as an HTTP server for production — any language,
              any client, standard API. Or import it directly as a Python
              library for scripts, notebooks, and rapid prototyping with
              zero network overhead.
            </p>
            <p>
              Same capabilities either way. Start with the library, graduate
              to the server when you need multi-language access or
              independent scaling.
            </p>
          </div>
          <div className={styles.serverComparison}>
            <div className={styles.comparisonRow}>
              <span className={styles.comparisonLabel}>Server</span>
              <code className={styles.comparisonCode}>POST /v1/responses</code>
              <span className={styles.comparisonGood}>any language</span>
            </div>
            <div className={styles.comparisonRow}>
              <span className={styles.comparisonLabel}>Library</span>
              <code className={styles.comparisonCode}>client.responses.create(...)</code>
              <span className={styles.comparisonGood}>zero overhead</span>
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function Providers() {
  return (
    <Section className={styles.providerSection}>
      <div className="container">
        <h2>23 inference providers. 13 vector stores. 7 safety backends.</h2>
        <p className={styles.providerDesc}>
          Develop locally with Ollama. Deploy to production with vLLM.
          Wrap Bedrock or Vertex without lock-in. Same API surface, different backend.
        </p>
        <div className={styles.providerGrid}>
          {PROVIDERS.map(p => (
            <Link key={p.name} className={styles.provider} to={p.href}>{p.name}</Link>
          ))}
        </div>
        <Link className={styles.textLink} to="/docs/providers">
          All providers
        </Link>
      </div>
    </Section>
  );
}

function Architecture() {
  return (
    <Section className={styles.archSection}>
      <div className="container">
        <h2>How it works</h2>
        <p className={styles.archDesc}>
          Your application talks to one process — either an HTTP server
          or an in-process library client. That process routes to pluggable
          providers for inference, vector storage, files, moderation, and tools.
          The composition happens at the OGX level, not in your application code.
        </p>
        <div className={styles.archImg}>
          <img src="/img/architecture-animated.svg" alt="OGX Architecture" loading="lazy" />
        </div>
      </div>
    </Section>
  );
}

function Bottom() {
  return (
    <Section className={styles.bottomSection}>
      <div className="container">
        <div className={styles.bottomLayout}>
          <div>
            <h2>Open source</h2>
            <p>
              Apache 2.0 licensed. Contributions welcome.
            </p>
          </div>
          <div className={styles.bottomLinks}>
            <a href="https://github.com/ogx-ai/ogx" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
            <a href="https://discord.gg/ZAFjsrcw" target="_blank" rel="noopener noreferrer">
              Discord
            </a>
            <Link to="/docs/">
              Documentation
            </Link>
            <Link to="/blog">
              Blog
            </Link>
          </div>
        </div>
      </div>
    </Section>
  );
}

export default function Home() {
  return (
    <Layout title="The Open-Source AI Application Server & Library" description="Inference, vector stores, moderation, tools, and agentic orchestration. Server or Python library, OpenAI + Anthropic + Google compatible, pluggable providers.">
      <main>
        <AnnouncementBanner />
        <Hero />
        <CliShowcase />
        <ApiSurface />
        <ServerAndLibrary />
        <Providers />
        <Architecture />
        <Bottom />
      </main>
    </Layout>
  );
}
