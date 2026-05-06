# OpenResponses Conformance Recordings

This directory holds inference recordings used by the
[OpenResponses conformance CI job](../../.github/workflows/openresponses-conformance.yml)
to replay underlying OpenAI API calls without needing a live API key.

## How recordings work

The ogx server intercepts outbound inference calls (e.g., to OpenAI) and
stores each request/response pair as a JSON file named by the SHA-256 hash of the
normalized request. In CI the server runs with `OGX_TEST_INFERENCE_MODE=replay`
and looks up those files instead of making real network calls.

## Adding or updating recordings

Use the provided script — it handles everything (installing deps, starting the
server, running the tests, and reporting what was written):

```bash
OPENAI_API_KEY=<your-key> bash scripts/record-openresponses-conformance.sh
```

Then commit the results:

```bash
git add tests/integration/openresponses/recordings/
git commit -m "chore: update OpenResponses conformance recordings"
```

## Known conformance gaps

See [`CONFORMANCE_GAPS.md`](./CONFORMANCE_GAPS.md) for a full breakdown of why
tests currently fail and what needs to change in the ogx implementation.
