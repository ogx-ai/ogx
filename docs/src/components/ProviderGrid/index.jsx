import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

const ICONS = {
  inference: (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2"/>
      <rect x="9" y="9" width="6" height="6"/>
      <path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>
      <path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>
    </svg>
  ),
  vector_io: (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>
      <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/>
    </svg>
  ),
  files: (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V7z"/>
      <path d="M14 2v4a2 2 0 002 2h4"/><path d="M10 13H8"/><path d="M16 17H8"/><path d="M16 13h-2"/>
    </svg>
  ),
  external: (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a6 6 0 01-12 0V8z"/>
    </svg>
  ),
};

const CATEGORIES = [
  {
    id: 'inference',
    label: 'Inference',
    count: 23,
    desc: 'Ollama, vLLM, OpenAI, Bedrock, Anthropic, Gemini, WatsonX, and more',
    href: '/docs/providers/inference/',
  },
  {
    id: 'vector_io',
    label: 'Vector IO',
    count: 15,
    desc: 'FAISS, SQLite-Vec, ChromaDB, Qdrant, Milvus, PGVector, Weaviate',
    href: '/docs/providers/vector_io/',
  },
  {
    id: 'files',
    label: 'Files',
    count: 3,
    desc: 'Local filesystem and S3 storage backends',
    href: '/docs/providers/files/',
  },
];

const EXTERNAL = {
  id: 'external',
  label: 'External Providers',
  desc: 'Build your own provider and integrate it with OGX',
  href: '/docs/providers/external/',
};

export function ProviderGrid() {
  return (
    <div className={styles.grid}>
      {CATEGORIES.map((cat) => (
        <Link key={cat.id} to={cat.href} className={styles.card}>
          <div className={styles.cardHeader}>
            <div className={styles.cardIcon}>{ICONS[cat.id]}</div>
            <span className={styles.cardTitle}>{cat.label}</span>
            <span className={styles.cardCount}>{cat.count}</span>
          </div>
          <p className={styles.cardDesc}>{cat.desc}</p>
        </Link>
      ))}
      <Link to={EXTERNAL.href} className={`${styles.card} ${styles.cardExternal}`}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}>{ICONS[EXTERNAL.id]}</div>
          <span className={styles.cardTitle}>{EXTERNAL.label}</span>
        </div>
        <p className={styles.cardDesc}>{EXTERNAL.desc}</p>
      </Link>
    </div>
  );
}

const REF_DATA = [
  { id: 'inference', label: 'Inference', count: 23, examples: 'Ollama, vLLM, OpenAI, Bedrock, Anthropic, Gemini, WatsonX, and more', href: '/docs/providers/inference/', linkLabel: 'Inference Providers' },
  { id: 'vector_io', label: 'Vector IO', count: 15, examples: 'FAISS, ChromaDB, Qdrant, Milvus, PGVector, Weaviate, Elasticsearch', href: '/docs/providers/vector_io/', linkLabel: 'Vector IO Providers' },
  { id: 'files', label: 'Files', count: 3, examples: 'Local filesystem, S3, OpenAI Files', href: '/docs/providers/files/', linkLabel: 'Files Providers' },
  { id: 'external', label: 'External', count: null, examples: 'Build your own provider', href: '/docs/providers/external/', linkLabel: 'External Providers Guide' },
];

export function ProviderReference() {
  return (
    <div className={styles.reference}>
      <table className={styles.refTable}>
        <thead>
          <tr>
            <th>Category</th>
            <th style={{textAlign: 'center'}}>Count</th>
            <th>Examples</th>
            <th>Docs</th>
          </tr>
        </thead>
        <tbody>
          {REF_DATA.map((row) => (
            <tr key={row.id} className={styles.refRow}>
              <td>
                <div className={styles.refCategory}>
                  <span className={styles.refCategoryIcon}>{ICONS[row.id]}</span>
                  {row.label}
                </div>
              </td>
              <td className={styles.refCount}>
                {row.count != null ? row.count : '—'}
              </td>
              <td className={styles.refExamples}>{row.examples}</td>
              <td className={styles.refLink}>
                <Link to={row.href}>
                  {row.linkLabel}
                  <span className={styles.refArrow}>&rarr;</span>
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
