import React, {useState, useEffect} from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

const GITHUB_RELEASES_URL =
  'https://api.github.com/repos/ogx-ai/ogx/releases?per_page=100';

const DOCS_BASE = 'https://ogx-ai.github.io';

const ARCHIVED_VERSIONS_URL = `${DOCS_BASE}/versionsArchived.json`;

function VersionsPage() {
  const [releases, setReleases] = useState([]);
  const [archivedVersions, setArchivedVersions] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [releasesResult, archivedResult] = await Promise.allSettled([
          fetchAllReleases(),
          fetch(ARCHIVED_VERSIONS_URL).then((r) =>
            r.ok ? r.json() : Promise.reject(new Error('not found'))
          ),
        ]);

        if (releasesResult.status === 'fulfilled') {
          setReleases(releasesResult.value);
        } else {
          setError(releasesResult.reason.message);
        }

        if (archivedResult.status === 'fulfilled') {
          setArchivedVersions(archivedResult.value);
        }
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const allReleases = releases;

  return (
    <Layout title="All Versions" description="All versions of OGX documentation">
      <main style={{maxWidth: 800, margin: '60px auto', padding: '0 24px'}}>
        <h1>OGX Docs</h1>
        <p style={{color: 'var(--ifm-color-secondary-darkest)', marginBottom: 32}}>
          All versions of the OGX documentation.
        </p>

        {loading && <p>Loading versions...</p>}

        {error && (
          <p style={{color: 'var(--ifm-color-danger)'}}>
            Failed to load versions: {error}
          </p>
        )}

        <div
          style={{
            background: 'var(--ifm-color-primary-lightest)',
            border: '1px solid var(--ifm-color-primary-light)',
            borderRadius: 8,
            padding: '16px 20px',
            marginBottom: 32,
          }}
        >
          <div
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              color: 'var(--ifm-color-primary)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: 4,
            }}
          >
            Next
          </div>
          <Link to="/" style={{fontSize: '1.1rem', fontWeight: 600}}>
            Current documentation
          </Link>
          <span
            style={{
              marginLeft: 12,
              fontSize: '0.85rem',
              color: 'var(--ifm-color-secondary-darkest)',
            }}
          >
            unreleased
          </span>
        </div>

        {allReleases.length > 0 && (
          <>
            <h2
              style={{
                fontSize: '1.1rem',
                color: 'var(--ifm-color-secondary-darkest)',
                marginBottom: 12,
              }}
            >
              Releases
            </h2>
            <ul style={{listStyle: 'none', padding: 0, margin: 0}}>
              {allReleases.map((release) => {
                const docsUrl = archivedVersions[release.tag_name];
                return (
                  <li
                    key={release.id}
                    style={{
                      padding: '10px 0',
                      borderBottom: '1px solid var(--ifm-color-emphasis-200)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                    }}
                  >
                    <a
                      href={release.html_url}
                      style={{
                        fontSize: '0.95rem',
                        fontWeight: 500,
                      }}
                    >
                      {release.tag_name}
                    </a>
                    <span
                      style={{
                        fontSize: '0.8rem',
                        color: 'var(--ifm-color-secondary-darkest)',
                      }}
                    >
                      {new Date(release.published_at).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                    <span style={{marginLeft: 'auto', display: 'flex', gap: 12}}>
                      {docsUrl && (
                        <a
                          href={docsUrl}
                          style={{
                            fontSize: '0.8rem',
                            color: 'var(--ifm-color-primary)',
                          }}
                        >
                          Documentation
                        </a>
                      )}
                    </span>
                  </li>
                );
              })}
            </ul>
          </>
        )}

        <div style={{marginTop: 32}}>
          <Link to="/" style={{fontSize: '0.9rem'}}>
            &larr; Back to docs
          </Link>
        </div>
      </main>
    </Layout>
  );
}

async function fetchAllReleases() {
  const all = [];
  let page = 1;
  let hasMore = true;
  while (hasMore) {
    const res = await fetch(`${GITHUB_RELEASES_URL}&page=${page}`);
    if (!res.ok) throw new Error(`GitHub API returned ${res.status}`);
    const data = await res.json();
    if (data.length === 0) {
      hasMore = false;
    } else {
      all.push(...data);
      page++;
    }
  }
  return all.filter((r) => !r.prerelease && !r.draft);
}

export default VersionsPage;
