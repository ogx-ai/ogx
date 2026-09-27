# OGX Technical Whitepaper (LaTeX)

This directory holds `ogx.tex` (and its build artifacts `ogx.bbl`, `ogx.pdf`,
`references.bib`), a standalone, diagram-heavy technical whitepaper. **It is a
separate document from the JOSS submission**, not an alternate build of it.

- **JOSS manuscript:** [`paper.md`](https://github.com/ogx-ai/ogx/blob/5f0d8b975da04364c1687b3cf3215d25da505201/paper.md)
  and [`paper.bib`](https://github.com/ogx-ai/ogx/blob/5f0d8b975da04364c1687b3cf3215d25da505201/paper.bib)
  at commit `5f0d8b975da04364c1687b3cf3215d25da505201`, including the
  corrected portability and moderation claims, release-scoped provider
  counts, API stability wording, and bibliography corrections. These links
  identify the manuscript independently of the
  software release tag.
- **This whitepaper:** `ogx.tex`, built against `references.bib` and `ogx.bbl`
  in this directory. It shares subject matter with `paper.md` but is
  maintained independently, is not kept in lockstep with it, and is not part
  of the JOSS submission.

Both documents currently share the same title, which has caused confusion
about which one the JOSS proof corresponds to. If you're looking for the JOSS
manuscript, use the pinned links above.

## Source references

The software version for this revision, operator source reference, and revised
manuscript are identified independently:

| Component | Revision | Reference |
| --- | --- | --- |
| OGX software | `v1.0.3` | commit `5393c94b2d23a3069b3708f9ca81ad350d2deb21` |
| OGX Kubernetes Operator | `v0.10.0` | commit `7fa16532e1434bf74493ca305b1e21030914ae57` |
| Manuscript (`paper.md` / `paper.bib`) | -- | commit `5f0d8b975da04364c1687b3cf3215d25da505201` |

The operator reference is an existing tagged source snapshot. Its
[`OGXServer` API](https://github.com/ogx-ai/ogx-k8s-operator/blob/7fa16532e1434bf74493ca305b1e21030914ae57/api/v1beta1/ogxserver_types.go)
and [deployment documentation](https://github.com/ogx-ai/ogx-k8s-operator/blob/7fa16532e1434bf74493ca305b1e21030914ae57/README.md)
describe the custom resource, network policies, ConfigMap image overrides,
Kubernetes/OpenShift deployment, and multi-architecture builds discussed in
the manuscript. This citation identifies source for those features; it does
not establish the operator version used by historical deployments.

## Release scope and registry counts

The manuscript describes the cited OGX `v1.0.3` source snapshot. The
[14 September review](https://github.com/openjournals/joss-reviews/issues/11234#issuecomment-5668826998)
inspected a newer main revision. Both sets of counts can be reproduced from
their registry files:

| Source snapshot | Inference provider types | Vector I/O provider types | Distinct vector backends |
| --- | --- | --- | --- |
| `v1.0.3` (`5393c94b`) | [23](https://github.com/ogx-ai/ogx/blob/5393c94b2d23a3069b3708f9ca81ad350d2deb21/src/ogx/providers/registry/inference.py) | [13](https://github.com/ogx-ai/ogx/blob/5393c94b2d23a3069b3708f9ca81ad350d2deb21/src/ogx/providers/registry/vector_io.py) | 10 |
| Reviewed main (`27c9a81`) | [25](https://github.com/ogx-ai/ogx/blob/27c9a810758f3b97b8775462be00fa9d4e9766f0/src/ogx/providers/registry/inference.py) | [14](https://github.com/ogx-ai/ogx/blob/27c9a810758f3b97b8775462be00fa9d4e9766f0/src/ogx/providers/registry/vector_io.py) | 11 |

Provider types count distinct registered `provider_type` strings. Vector
backends count distinct names after removing the `inline::` or `remote::`
prefix. In `v1.0.3`, these names are `chromadb`, `elasticsearch`, `faiss`,
`infinispan`, `milvus`, `oci`, `pgvector`, `qdrant`, `sqlite-vec`, and
`weaviate`. ChromaDB, Milvus, and Qdrant each have inline and remote variants.
The reviewed main revision adds the `neo4j` backend.

Neither snapshot has a Safety or Shields provider protocol. Both implement
optional Responses guardrails using an external moderation endpoint. The
[`v1.0.3` configuration](https://github.com/ogx-ai/ogx/blob/5393c94b2d23a3069b3708f9ca81ad350d2deb21/src/ogx/providers/inline/responses/builtin/config.py)
exposes `moderation_endpoint`; the newer main documentation also describes
`moderation_headers`.

Interactions is experimental at `/v1alpha/interactions` in the cited release.
Containers and Skills are absent from that release and are omitted from the
manuscript. Their routes in reviewed main are `/v1alpha/containers` and
`/v1alpha/skills`, respectively.

The [reference audit](reference-audit.md) records checks of the 19 entries
without DOIs and the verified bibliography corrections.

## JOSS proof

The [JOSS proof generated on 10 September 2026 at 20:51 UTC](https://github.com/openjournals/joss-papers/blob/86bf0fe916ed4c2c753cf90221f91b78a1efb143/joss.11234/10.21105.joss.11234.pdf)
contains the shortened manuscript, revised deployment wording, OGX `v1.0.3`
source citation, versioned operator citation, and corrected SGLang reference.
Its corresponding manuscript pair is pinned at
[`7ce8d77a4faa98863529f227d874b75770b6c805`](https://github.com/ogx-ai/ogx/tree/7ce8d77a4faa98863529f227d874b75770b6c805);
those files are byte-identical to the merged source at
[`016a6079594de71b58cba2e58842126fadda2ec1`](https://github.com/ogx-ai/ogx/tree/016a6079594de71b58cba2e58842126fadda2ec1)
when the proof was requested. This identifies a source snapshot, without
claiming the bot's checkout commit. The PDF SHA-256 is
`4e076a5036c6c6efd7c97536411e58f515d35f604f6de3fecdc124af458bd99e`.

The rendered proof includes the AI disclosure on page 4, immutable OGX and
operator source references on page 5, and the corrected SGLang author list
and DOI on page 6.

That proof predates the portability, moderation, provider-count, API stability,
and bibliography corrections pinned above. After merging those updates, run
`@editorialbot generate pdf` on the
[JOSS review issue](https://github.com/openjournals/joss-reviews/issues/11234)
to generate a proof with the revised manuscript, and
record the new proof link and hash alongside its manuscript source revision. If either
manuscript file changes, update both source links and the table before
regenerating the proof.
