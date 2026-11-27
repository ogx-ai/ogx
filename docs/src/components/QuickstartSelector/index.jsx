import React, {useEffect, useState} from 'react';
import Admonition from '@theme/Admonition';
import CodeBlock from '@theme/CodeBlock';
import ReactMarkdown from 'react-markdown';
import styles from './styles.module.css';

// Steps and options come from docs/docs/getting_started/quickstart/steps.json, which
// scripts/quickstart_smoke.py runs in CI. Keep the selection rules below in sync with it.

function matches(when, selection) {
  if (!when) return true;
  return Object.entries(when).every(([dimension, values]) => values.includes(selection[dimension]));
}

// Drop dimensions that do not apply to the selection and fill in the rest with defaults.
function normalize(dimensions, requested) {
  const selection = {};
  for (const dimension of dimensions) {
    if (!matches(dimension.when, selection)) continue;
    const ids = dimension.options.map((option) => option.id);
    selection[dimension.id] = ids.includes(requested[dimension.id]) ? requested[dimension.id] : ids[0];
  }
  return selection;
}

function unsupportedReason(manifest, selection) {
  const rule = manifest.unsupported.find((candidate) => matches(candidate.when, selection));
  return rule ? rule.reason : null;
}

function modelFor(manifest, selection) {
  const backend = manifest.dimensions.find((dimension) => dimension.id === 'backend');
  return backend.options.find((option) => option.id === selection.backend).model;
}

function codeLines(step, selection) {
  return (step.code || [])
    .filter((entry) => typeof entry === 'string' || matches(entry.when, selection))
    .map((entry) => (typeof entry === 'string' ? entry : entry.line));
}

// Snippet files start with the repository license header, which readers do not need.
function stripLicenseHeader(source) {
  const lines = source.split('\n');
  if (!lines[0].startsWith('# Copyright')) return source;
  const firstBlank = lines.findIndex((line) => line.trim() === '');
  return lines.slice(firstBlank + 1).join('\n');
}

const markdownComponents = {
  pre: ({children}) => <>{children}</>,
  code: ({className, children}) =>
    className ? (
      <CodeBlock language={className.replace('language-', '')}>{String(children).replace(/\n$/, '')}</CodeBlock>
    ) : (
      <code>{children}</code>
    ),
};

function StepText({text, model}) {
  return (
    <div className={styles.stepText}>
      <ReactMarkdown components={markdownComponents}>{text.replaceAll('{model}', model)}</ReactMarkdown>
    </div>
  );
}

function StepCode({step, selection, files, manifest, model}) {
  const swapModel = (code) => code.replaceAll(manifest.default_model, model);
  if (step.file) {
    return (
      <CodeBlock language={step.language} title={step.file}>
        {swapModel(stripLicenseHeader(files[step.file])).trimEnd()}
      </CodeBlock>
    );
  }
  const lines = codeLines(step, selection);
  if (lines.length === 0) return null;
  return <CodeBlock language={step.language}>{swapModel(lines.join('\n'))}</CodeBlock>;
}

// A titled step starts a numbered section; untitled steps continue the one before.
function sections(steps) {
  const result = [];
  for (const step of steps) {
    if (step.title || result.length === 0) result.push({title: step.title, steps: [step]});
    else result[result.length - 1].steps.push(step);
  }
  return result;
}

function readSelectionFromUrl(dimensions) {
  const params = new URLSearchParams(window.location.search);
  return normalize(dimensions, Object.fromEntries(dimensions.map((d) => [d.id, params.get(d.id)])));
}

function writeSelectionToUrl(selection) {
  const url = new URL(window.location.href);
  for (const key of [...url.searchParams.keys()]) url.searchParams.delete(key);
  for (const [key, value] of Object.entries(selection)) url.searchParams.set(key, value);
  window.history.replaceState(window.history.state, '', url);
}

export default function QuickstartSelector({manifest, files}) {
  const {dimensions} = manifest;
  const [selection, setSelection] = useState(() => normalize(dimensions, {}));

  useEffect(() => {
    setSelection(readSelectionFromUrl(dimensions));
  }, [dimensions]);

  const choose = (dimensionId, optionId) => {
    const next = normalize(dimensions, {...selection, [dimensionId]: optionId});
    setSelection(next);
    writeSelectionToUrl(next);
  };

  const reason = unsupportedReason(manifest, selection);
  const model = modelFor(manifest, selection);
  const steps = manifest.steps.filter((step) => matches(step.when, selection));

  return (
    <div className={styles.selector}>
      <div className={styles.dimensions}>
        {dimensions
          .filter((dimension) => matches(dimension.when, selection))
          .map((dimension) => (
            <div className={styles.dimension} key={dimension.id}>
              <span className={styles.dimensionLabel} id={`quickstart-${dimension.id}`}>
                {dimension.label}
              </span>
              <div className={styles.options} role="radiogroup" aria-labelledby={`quickstart-${dimension.id}`}>
                {dimension.options.map((option) => {
                  const active = selection[dimension.id] === option.id;
                  const blocked = unsupportedReason(manifest, normalize(dimensions, {...selection, [dimension.id]: option.id}));
                  return (
                    <button
                      key={option.id}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      title={blocked && !active ? 'Not available with the current selection' : undefined}
                      className={[styles.option, active && styles.optionActive, blocked && styles.optionBlocked]
                        .filter(Boolean)
                        .join(' ')}
                      onClick={() => choose(dimension.id, option.id)}>
                      {option.label}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
      </div>

      {reason ? (
        <Admonition type="info" title="Not available with this combination">
          <ReactMarkdown>{reason}</ReactMarkdown>
        </Admonition>
      ) : (
        <ol className={styles.steps}>
          {sections(steps).map((section) => (
            <li className={styles.step} key={section.steps[0].id}>
              {section.title && <h3 className={styles.stepTitle}>{section.title}</h3>}
              {section.steps.map((step) => (
                <React.Fragment key={step.id}>
                  {step.text && <StepText text={step.text} model={model} />}
                  <StepCode step={step} selection={selection} files={files} manifest={manifest} model={model} />
                </React.Fragment>
              ))}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
