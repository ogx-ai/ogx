// Copyright (c) The OGX Contributors.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

/**
 * Global setup for integration tests.
 * This file mimics pytest's fixture system by providing shared test configuration.
 */

import LlamaStackClient from 'llama-stack-client';

/**
 * Load test configuration from the Python setup system.
 * This reads setup definitions from tests/integration/suites.py via get_setup_env.py.
 */
function loadTestConfig() {
  const baseURL = process.env['TEST_API_BASE_URL'];
  const setupName = process.env['OGX_TEST_SETUP'];
  const textModel = process.env['OGX_TEST_TEXT_MODEL'];
  const embeddingModel = process.env['OGX_TEST_EMBEDDING_MODEL'];

  if (!baseURL) {
    throw new Error(
      'TEST_API_BASE_URL is required for integration tests. ' +
        'Run tests using: ./scripts/integration-test.sh',
    );
  }

  return {
    baseURL,
    textModel,
    embeddingModel,
    setupName,
  };
}

// Read configuration from environment variables (set by scripts/integration-test.sh)
export const TEST_CONFIG = loadTestConfig();

// Validate required configuration
beforeAll(() => {
  console.log('\n=== Integration Test Configuration ===');
  console.log(`Base URL: ${TEST_CONFIG.baseURL}`);
  console.log(`Setup: ${TEST_CONFIG.setupName || 'NOT SET'}`);
  console.log(
    `Text Model: ${TEST_CONFIG.textModel || 'NOT SET - tests requiring text model will be skipped'}`,
  );
  console.log(
    `Embedding Model: ${
      TEST_CONFIG.embeddingModel || 'NOT SET - tests requiring embedding model will be skipped'
    }`,
  );
  console.log('=====================================\n');
});

/**
 * Create a client instance for integration tests.
 * Mimics pytest's `ogx_client` fixture.
 *
 * @param testId - Test ID to send in X-OGX-Provider-Data header for replay mode.
 *                 Format: "tests/integration/responses/test_basic_responses.py::test_name[params]"
 */
export function createTestClient(testId?: string): LlamaStackClient {
  const headers: Record<string, string> = {};

  // In server mode with replay, send test ID for recording isolation
  if (process.env['OGX_TEST_STACK_CONFIG_TYPE'] === 'server' && testId) {
    headers['X-OGX-Provider-Data'] = JSON.stringify({
      __test_id: testId,
    });
  }

  return new LlamaStackClient({
    baseURL: TEST_CONFIG.baseURL,
    timeout: 60000, // 60 seconds
    defaultHeaders: headers,
  });
}

/**
 * Skip test if required model is not configured.
 * Mimics pytest's `skip_if_no_model` autouse fixture.
 */
export function skipIfNoModel(modelType: 'text' | 'embedding'): typeof test {
  const model = modelType === 'text' ? TEST_CONFIG.textModel : TEST_CONFIG.embeddingModel;

  if (!model) {
    const envVar = modelType === 'text' ? 'OGX_TEST_TEXT_MODEL' : 'OGX_TEST_EMBEDDING_MODEL';
    const message = `Skipping: ${modelType} model not configured (set ${envVar})`;
    return test.skip.bind(test) as typeof test;
  }

  return test;
}

/**
 * Get the configured text model, throwing if not set.
 * Use this in tests that absolutely require a text model.
 */
export function requireTextModel(): string {
  if (!TEST_CONFIG.textModel) {
    throw new Error(
      'OGX_TEST_TEXT_MODEL environment variable is required. ' +
        'Run tests using: ./scripts/integration-test.sh',
    );
  }
  return TEST_CONFIG.textModel;
}

/**
 * Get the configured embedding model, throwing if not set.
 * Use this in tests that absolutely require an embedding model.
 */
export function requireEmbeddingModel(): string {
  if (!TEST_CONFIG.embeddingModel) {
    throw new Error(
      'OGX_TEST_EMBEDDING_MODEL environment variable is required. ' +
        'Run tests using: ./scripts/integration-test.sh',
    );
  }
  return TEST_CONFIG.embeddingModel;
}

/**
 * Extracts aggregated text output from a ResponseObject.
 * This concatenates all text content from the response's output array.
 *
 * Copied from ogx-client's response-helpers until it's available in published version.
 */
export function getResponseOutputText(response: any): string {
  const pieces: string[] = [];

  for (const output of response.output ?? []) {
    if (!output || output.type !== 'message') {
      continue;
    }

    const content = output.content;
    if (typeof content === 'string') {
      pieces.push(content);
      continue;
    }

    if (!Array.isArray(content)) {
      continue;
    }

    for (const item of content) {
      if (typeof item === 'string') {
        pieces.push(item);
        continue;
      }
      if (item && item.type === 'output_text' && 'text' in item && typeof item.text === 'string') {
        pieces.push(item.text);
      }
    }
  }

  return pieces.join('');
}
