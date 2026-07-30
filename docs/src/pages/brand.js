import React from 'react';
import Layout from '@theme/Layout';
import styles from './brand.module.css';

function LogoCell({src, label, variant}) {
  const cellClass = variant === 'dark' ? styles.logoCellDark : styles.logoCellLight;
  return (
    <div className={`${styles.logoCell} ${cellClass}`}>
      <span className={styles.logoCellLabel}>{label}</span>
      <img src={src} alt={`OGX logo – ${label}`} width="180" height="180" />
    </div>
  );
}

function LockupCell({src, label, variant}) {
  const cellClass = variant === 'dark' ? styles.lockupCellDark : styles.lockupCellLight;
  return (
    <div className={`${styles.lockupCell} ${cellClass}`}>
      <span className={styles.logoCellLabel}>{label}</span>
      <img src={src} alt={`OGX lockup – ${label}`} style={{maxWidth: 340, width: '100%'}} />
    </div>
  );
}

function Swatch({hex, name, style}) {
  return (
    <div className={styles.swatch} style={style}>
      <span className={styles.swatchHex}>{hex}</span>
      <span className={styles.swatchName}>{name}</span>
    </div>
  );
}

function SizePreview({size}) {
  return (
    <div className={styles.sizeItem}>
      <img src="/img/brand/ogx-icon-color.svg" alt={`${size}px`} width={size} height={size} />
      <span>{size}px</span>
    </div>
  );
}

function DownloadCard({href, name, desc}) {
  return (
    <a className={styles.downloadCard} href={href} download>
      <strong>{name}</strong>
      <span>{desc}</span>
    </a>
  );
}

export default function BrandPage() {
  return (
    <Layout title="Brand" description="OGX brand assets, logo variants, and usage guidelines.">
      <main className={styles.brandPage}>
        <div className={styles.container}>

          {/* Header */}
          <div className={styles.header}>
            <h1>Brand</h1>
            <p>
              Logo assets, color palette, and usage guidelines for the OGX mark.
              All assets are SVG and free to use when referencing the project.
            </p>
          </div>

          {/* Logo Mark */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Logo Mark</h2>
            <p className={styles.sectionDesc}>
              The OGX mark is built from five graduated diamond tiles arranged in a beacon
              formation. The top tile is smallest, the bottom tiles largest — representing a
              signal rising from a stable foundation.
            </p>
            <div className={styles.logoGrid}>
              <LogoCell src="/img/brand/ogx-icon-color.svg" label="Color on white" variant="light" />
              <LogoCell src="/img/brand/ogx-icon-color.svg" label="Color on dark" variant="dark" />
              <LogoCell src="/img/brand/ogx-icon-mono.svg" label="Mono on white" variant="light" />
              <LogoCell src="/img/brand/ogx-icon-reverse.svg" label="Reverse on dark" variant="dark" />
            </div>
          </section>

          {/* Lockup */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Lockup</h2>
            <p className={styles.sectionDesc}>
              The horizontal lockup pairs the mark with the OGX wordmark in Archivo 800.
              Use the color version on light backgrounds and the reverse version on dark.
            </p>
            <div className={styles.lockupRow}>
              <LockupCell src="/img/brand/ogx-lockup-color.svg" label="Color lockup" variant="light" />
              <LockupCell src="/img/brand/ogx-lockup-reverse.svg" label="Reverse lockup" variant="dark" />
            </div>
          </section>

          {/* Sizes */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Minimum Sizes</h2>
            <p className={styles.sectionDesc}>
              The mark is readable down to 24px. Below that, use the favicon.
            </p>
            <div className={styles.sizesRow}>
              <SizePreview size={24} />
              <SizePreview size={32} />
              <SizePreview size={48} />
              <SizePreview size={64} />
              <SizePreview size={96} />
              <SizePreview size={128} />
            </div>
          </section>

          {/* Palette */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Color Palette</h2>
            <p className={styles.sectionDesc}>
              The brand palette is derived from the documentation site and uses a teal/navy system.
            </p>
            <div className={styles.paletteGrid}>
              <Swatch hex="#2DBDC2" name="Bright Teal" style={{background: '#2DBDC2'}} />
              <Swatch hex="#0D7377" name="Deep Teal" style={{background: '#0D7377'}} />
              <Swatch hex="#1E1E2E" name="Navy" style={{background: '#1E1E2E'}} />
              <Swatch hex="#68717B" name="Slate" style={{background: '#68717B'}} />
            </div>
          </section>

          {/* Guidelines */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Usage Guidelines</h2>
            <p className={styles.sectionDesc}>
              Keep the mark clear and recognizable across all contexts.
            </p>
            <div className={styles.guidelineGrid}>
              <div className={`${styles.guidelineItem} ${styles.guidelineDo}`}>
                <div className={styles.guidelineLabel}>Do</div>
                <p>Use the color version on light backgrounds and the reverse version on dark backgrounds.</p>
              </div>
              <div className={`${styles.guidelineItem} ${styles.guidelineDo}`}>
                <div className={styles.guidelineLabel}>Do</div>
                <p>Maintain clear space around the mark equal to the height of the top tile.</p>
              </div>
              <div className={`${styles.guidelineItem} ${styles.guidelineDont}`}>
                <div className={styles.guidelineLabel}>Don't</div>
                <p>Rotate, stretch, or rearrange the tiles. The beacon formation is fixed.</p>
              </div>
              <div className={`${styles.guidelineItem} ${styles.guidelineDont}`}>
                <div className={styles.guidelineLabel}>Don't</div>
                <p>Place the color version on busy backgrounds where tiles lose contrast.</p>
              </div>
              <div className={`${styles.guidelineItem} ${styles.guidelineDo}`}>
                <div className={styles.guidelineLabel}>Do</div>
                <p>Use the monochrome version when color reproduction is limited (print, embossing).</p>
              </div>
              <div className={`${styles.guidelineItem} ${styles.guidelineDont}`}>
                <div className={styles.guidelineLabel}>Don't</div>
                <p>Add drop shadows, gradients, outlines, or other effects to the mark.</p>
              </div>
            </div>
          </section>

          {/* Downloads */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Download</h2>
            <p className={styles.sectionDesc}>
              All assets are SVG. Right-click or tap to save.
            </p>
            <div className={styles.downloadGrid}>
              <DownloadCard href="/img/brand/ogx-icon-color.svg" name="Icon — Color" desc="SVG, 512×512" />
              <DownloadCard href="/img/brand/ogx-icon-mono.svg" name="Icon — Mono" desc="SVG, 512×512" />
              <DownloadCard href="/img/brand/ogx-icon-reverse.svg" name="Icon — Reverse" desc="SVG, 512×512" />
              <DownloadCard href="/img/brand/ogx-lockup-color.svg" name="Lockup — Color" desc="SVG, 460×160" />
              <DownloadCard href="/img/brand/ogx-lockup-mono.svg" name="Lockup — Mono" desc="SVG, 460×160" />
              <DownloadCard href="/img/brand/ogx-lockup-reverse.svg" name="Lockup — Reverse" desc="SVG, 460×160" />
            </div>
          </section>

        </div>
      </main>
    </Layout>
  );
}
