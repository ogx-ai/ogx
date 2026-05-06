# Test Recording System

This directory contains recorded inference API responses used for deterministic testing without requiring live API access.

For more information, see the
[docs](https://ogx-ai.github.io/docs/contributing/testing/record-replay).
This README provides more technical information.

## Structure

- `responses/` - JSON files containing request/response pairs for inference operations

## Recording Format

Each JSON file contains:

- `request` - The normalized request parameters (method, endpoint, body)
- `response` - The response body (serialized from Pydantic models)

## Normalization

To reduce noise in git diffs, the recording system automatically normalizes fields that vary between runs but don't affect test behavior:

### OpenAI-style responses

- `id` - Deterministic hash based on request: `rec-{request_hash[:12]}`
- `created` - Normalized to epoch: `0`

### Ollama-style responses

- `created_at` - Normalized to: `"1970-01-01T00:00:00.000000Z"`
- `total_duration` - Normalized to: `0`
- `load_duration` - Normalized to: `0`
- `prompt_eval_duration` - Normalized to: `0`
- `eval_duration` - Normalized to: `0`

These normalizations ensure that re-recording tests produces minimal git diffs, making it easier to review actual changes to test behavior.

## Usage

### Replay mode (default)

Responses are replayed from recordings:

```bash
OGX_TEST_INFERENCE_MODE=replay pytest tests/integration/
```

### Record-if-missing mode (recommended for adding new tests)

Records only when no recording exists, otherwise replays. Use this for iterative development:

```bash
OGX_TEST_INFERENCE_MODE=record-if-missing pytest tests/integration/
```

### Recording mode

**Force-records all API interactions**, overwriting existing recordings. Use with caution:

```bash
OGX_TEST_INFERENCE_MODE=record pytest tests/integration/
```

### Live mode

Skip recordings entirely and use live APIs:

```bash
OGX_TEST_INFERENCE_MODE=live pytest tests/integration/
```

## Re-normalizing Existing Recordings

If you need to apply normalization to existing recordings (e.g., after updating the normalization logic):

```bash
python scripts/normalize_recordings.py
```

Use `--dry-run` to preview changes without modifying files.
