import React, {useEffect, useState} from 'react';
import styles from './styles.module.css';

const EXAMPLES = [
  {
    label: 'Local (Ollama)',
    command: "uvx --from 'ogx[starter]' ogx stack run starter",
    tokens: [
      { text: 'uvx', style: 'tokenBinary' },
      { text: '--from', style: 'tokenFlag' },
      { text: "'ogx[starter]'", style: 'tokenPackage' },
      { text: 'ogx', style: 'tokenCommand' },
      { text: 'stack', style: 'tokenSub' },
      { text: 'run', style: 'tokenSub' },
      { text: 'starter', style: 'tokenAccent' },
    ],
  },
  {
    label: 'OpenAI',
    command: "export OPENAI_API_KEY=sk-xxx\nuvx --from 'ogx[starter]' ogx stack run starter",
    lines: [
      [
        { text: 'export', style: 'tokenBinary' },
        { text: 'OPENAI_API_KEY=sk-xxx', style: 'tokenFlag' },
      ],
      [
        { text: 'uvx', style: 'tokenBinary' },
        { text: '--from', style: 'tokenFlag' },
        { text: "'ogx[starter]'", style: 'tokenPackage' },
        { text: 'ogx', style: 'tokenCommand' },
        { text: 'stack', style: 'tokenSub' },
        { text: 'run', style: 'tokenSub' },
        { text: 'starter', style: 'tokenAccent' },
      ],
    ],
  },
];

function CopyIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="10" height="10" rx="2" />
      <path d="M5 15V7a2 2 0 0 1 2-2h8" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
      <path d="m5 13 4 4L19 7" />
    </svg>
  );
}

export default function InstallBlock() {
  const [active, setActive] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(id);
  }, [copied]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(EXAMPLES[active].command);
      setCopied(true);
    } catch {
      /* fallback */
    }
  };

  return (
    <div className={styles.installBlock}>
      <p className={styles.tagline}>
        Try it now, no installation required{' '}
        <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank" rel="noopener noreferrer" className={styles.taglineLink}>(requires uv)</a>
      </p>
      <div className={styles.tabRow}>
        {EXAMPLES.map((ex, i) => (
          <button
            key={ex.label}
            type="button"
            className={`${styles.tab} ${i === active ? styles.tabActive : ''}`}
            onClick={() => { setActive(i); setCopied(false); }}
          >
            {ex.label}
          </button>
        ))}
      </div>
      <div className={styles.commandRow}>
        <code className={styles.command}>
          <span className={styles.commandReveal} key={active}>
            {EXAMPLES[active].lines ? (
              EXAMPLES[active].lines.map((line, li) => (
                <span key={li} className={styles.commandLine}>
                  {line.map((tok, ti) => (
                    <span key={ti}>
                      {ti > 0 && <span className={styles.space}> </span>}
                      <span className={styles[tok.style]}>{tok.text}</span>
                    </span>
                  ))}
                </span>
              ))
            ) : (
              EXAMPLES[active].tokens.map((tok, i) => (
                <span key={i}>
                  {i > 0 && <span className={styles.space}> </span>}
                  <span className={styles[tok.style]}>{tok.text}</span>
                </span>
              ))
            )}
          </span>
          <span className={styles.cursor} />
        </code>
        <button
          type="button"
          className={`${styles.copyBtn} ${copied ? styles.copyBtnCopied : ''}`}
          onClick={handleCopy}
          aria-label="Copy install command"
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
    </div>
  );
}
