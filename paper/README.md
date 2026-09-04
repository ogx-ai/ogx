# OGX Technical Whitepaper (LaTeX)

This directory holds `ogx.tex` (and its build artifacts `ogx.bbl`, `ogx.pdf`,
`references.bib`), a standalone, diagram-heavy technical whitepaper. **It is a
separate document from the JOSS submission**, not an alternate build of it.

- **JOSS manuscript:** [`/paper.md`](../paper.md) at the repository root, built
  against [`/paper.bib`](../paper.bib). This is the source reviewed by JOSS
  (openjournals/joss-reviews#11234) and the manuscript the generated proof PDF
  corresponds to.
- **This whitepaper:** `ogx.tex`, built against `references.bib` and `ogx.bbl`
  in this directory. It shares subject matter with `/paper.md` but is
  maintained independently, is not kept in lockstep with it, and is not part
  of the JOSS submission.

Both documents currently share the same title, which has caused confusion
about which one the JOSS proof corresponds to. If you're looking for the JOSS
manuscript, use `/paper.md`.
