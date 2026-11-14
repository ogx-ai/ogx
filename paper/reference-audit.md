# JOSS reference audit, 14 September 2026

The 22 bibliography entries contain three verified DOIs (`arceo2026securing`,
`kwon2023vllm`, and `sglang`). The other 19 entries were checked against their
official project, publisher, author, or vendor sources. No additional DOI was
verified for the exact cited works. This records the search result; it does not
assert that no identifier could ever be assigned or found.

| Entries | Cited object | Result |
| --- | --- | --- |
| `ogx`, `ogxk8soperator`, `llamastack` | Software repositories; the OGX and operator citations pin source revisions | No matching official archive DOI found |
| `langchain`, `langgraph`, `llamaindex`, `crewai`, `haystack`, `sqlitevec` | Software projects | No usable DOI verified from project citation metadata or matching DataCite records |
| `openresponses`, `openaiResponsesAPI`, `mcp`, `databricksAgentFramework` | Specifications, API reference, and product documentation | No DOI verified for the cited documentation |
| `ibm_rag_milvus`, `oracle_oci_ogx`, `redhat_ops_agent` | Vendor blog/tutorial pages | No DOI verified for the cited posts |
| `meta_connect_ogx`, `ibm_techxchange_ogx` | Conference session and recorded presentation | No DOI verified for the cited media pages |
| `mlflow` | 2018 IEEE Data Engineering Bulletin article | No exact DOI match found in Crossref or the author-hosted article |

Software lookups used official README/citation metadata and DataCite searches for
the repository URLs. Records for papers or datasets that merely mention a project
were excluded. In particular:

- [LangChain's citation metadata](https://github.com/langchain-ai/langchain/blob/master/CITATION.cff)
  and [Haystack's citation metadata](https://github.com/deepset-ai/haystack/blob/main/CITATION.cff)
  provide repository URLs without DOIs.
- LlamaIndex's citation metadata contains `10.5281/zenodo.1234`, but it does not
  resolve. The project's [issue #14810](https://github.com/run-llama/llama_index/issues/14810)
  documents this invalid identifier. It must not be copied into the bibliography.
- The [author-hosted MLflow article](https://people.eecs.berkeley.edu/~matei/papers/2018/ieee_mlflow.pdf)
  identifies the exact 2018 work and its authors. The
  [author's publication list](https://people.eecs.berkeley.edu/~matei/)
  supplies volume 41, issue 4. A later MLflow conference paper is a different
  work and its DOI would not identify this article.

Two metadata corrections are included in `paper.bib`: the MLflow entry uses the
author-hosted article, Matei Zaharia's published name, the full journal name, and
issue 4; the [Oracle post](https://blogs.oracle.com/ai-and-datascience/accelerating-enterprise-gen-ai-applications-development-on-oci-with-llama-stack-and-oci-ai-blueprints)
is attributed to Amar Gowda and Dennis Kennetz and dated 2026, matching its
21 January 2026 publication date.

The Databricks product URL redirects to Agent Bricks. Its
[March 2025 release notes](https://docs.databricks.com/aws/en/release-notes/product/2025/march)
independently record Mosaic AI Agent Framework's general availability. This
redirect does not supply a DOI for the original product reference.
