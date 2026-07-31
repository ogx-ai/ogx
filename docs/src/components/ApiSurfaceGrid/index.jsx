import React from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function ApiSurfaceGrid({sections}) {
  return (
    <div className={styles.wrapper}>
      {sections.map((section) => (
        <div key={section.title} className={styles.section}>
          <h2 className={styles.sectionTitle}>
            {section.title}
          </h2>
          <div className={styles.grid}>
            {section.items.map((item) => (
              <Link
                key={item.href}
                to={item.href}
                className={`${styles.card} ${styles[section.tone] || ''}`}
              >
                <code className={styles.endpoint}>{item.endpoint}</code>
                <h3 className={styles.cardTitle}>{item.title}</h3>
                <p className={styles.cardBlurb}>{item.blurb}</p>
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
