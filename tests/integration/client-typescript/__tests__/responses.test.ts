// Copyright (c) The OGX Contributors.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

/**
 * Integration tests for Responses API.
 * Ported from: ogx/tests/integration/responses/test_basic_responses.py
 *
 * IMPORTANT: Test cases and IDs must match EXACTLY with Python tests to use recorded API responses.
 */

import { createTestClient, requireTextModel, getResponseOutputText } from '../setup';

describe('Responses API - Basic', () => {
  // Test cases matching ogx/tests/integration/responses/fixtures/test_cases.py
  const basicTestCases = [
    {
      id: 'earth',
      input: 'Which planet do humans live on?',
      expected: 'earth',
      // Use client_with_models fixture to match non-streaming recordings
      testId:
        'tests/integration/responses/test_basic_responses.py::test_response_non_streaming_basic[client_with_models-txt=openai/gpt-4o-earth]',
    },
    {
      id: 'saturn',
      input: 'Which planet has rings around it with a name starting with letter S?',
      expected: 'saturn',
      testId:
        'tests/integration/responses/test_basic_responses.py::test_response_non_streaming_basic[client_with_models-txt=openai/gpt-4o-saturn]',
    },
  ];

  test.each(basicTestCases)('non-streaming basic response: $id', async ({ input, expected, testId }) => {
    // Create client with test_id for all requests
    const client = createTestClient(testId);
    const textModel = requireTextModel();

    // Create a response
    const response = await client.responses.create({
      model: textModel,
      input,
      stream: false,
    });

    // Verify response has content
    const outputText = getResponseOutputText(response).toLowerCase().trim();
    expect(outputText.length).toBeGreaterThan(0);
    expect(outputText).toContain(expected.toLowerCase());

    // Verify usage is reported
    expect(response.usage).toBeDefined();
    expect(response.usage!.input_tokens).toBeGreaterThan(0);
    expect(response.usage!.output_tokens).toBeGreaterThan(0);
    expect(response.usage!.total_tokens).toBe(response.usage!.input_tokens + response.usage!.output_tokens);

    // Verify stored response matches
    const retrievedResponse = await client.responses.retrieve(response.id);
    expect(getResponseOutputText(retrievedResponse)).toBe(getResponseOutputText(response));

    // Test follow-up with previous_response_id
    const nextResponse = await client.responses.create({
      model: textModel,
      input: 'Repeat your previous response in all caps.',
      previous_response_id: response.id,
    });
    const nextOutputText = getResponseOutputText(nextResponse).trim();
    expect(nextOutputText).toContain(expected.toUpperCase());
  });

  test.each(basicTestCases)('streaming basic response: $id', async ({ input, expected, testId }) => {
    // Modify test_id for streaming variant
    const streamingTestId = testId.replace(
      'test_response_non_streaming_basic',
      'test_response_streaming_basic',
    );
    const client = createTestClient(streamingTestId);
    const textModel = requireTextModel();

    // Create a streaming response
    const stream = await client.responses.create({
      model: textModel,
      input,
      stream: true,
    });

    const events: any[] = [];
    let responseId = '';

    for await (const chunk of stream) {
      events.push(chunk);

      if (chunk.type === 'response.created') {
        // Verify response.created is the first event
        expect(events.length).toBe(1);
        expect(chunk.response.status).toBe('in_progress');
        responseId = chunk.response.id;
      } else if (chunk.type === 'response.completed') {
        // Verify response.completed comes after response.created
        expect(events.length).toBeGreaterThanOrEqual(2);
        expect(chunk.response.status).toBe('completed');
        expect(chunk.response.id).toBe(responseId);

        // Verify content quality
        const outputText = getResponseOutputText(chunk.response).toLowerCase().trim();
        expect(outputText.length).toBeGreaterThan(0);
        expect(outputText).toContain(expected.toLowerCase());

        // Verify usage is reported
        expect(chunk.response.usage).toBeDefined();
        expect(chunk.response.usage!.input_tokens).toBeGreaterThan(0);
        expect(chunk.response.usage!.output_tokens).toBeGreaterThan(0);
        expect(chunk.response.usage!.total_tokens).toBe(
          chunk.response.usage!.input_tokens + chunk.response.usage!.output_tokens,
        );
      }
    }

    // Verify we got both events
    expect(events.length).toBeGreaterThanOrEqual(2);
    const firstEvent = events[0];
    const lastEvent = events[events.length - 1];
    expect(firstEvent.type).toBe('response.created');
    expect(lastEvent.type).toBe('response.completed');

    // Verify stored response matches streamed response
    const retrievedResponse = await client.responses.retrieve(responseId);
    expect(getResponseOutputText(retrievedResponse)).toBe(getResponseOutputText(lastEvent.response));
  });
});
