// Copyright (c) The OGX Contributors.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

/**
 * Integration tests for Inference API (Chat Completions).
 * Ported from: ogx/tests/integration/inference/test_openai_completion.py
 *
 * IMPORTANT: Test cases must match EXACTLY with Python tests to use recorded API responses.
 */

import { createTestClient, requireTextModel } from '../setup';

describe('Inference API - Chat Completions', () => {
  // Test cases matching ogx/tests/integration/test_cases/inference/chat_completion.json
  const chatCompletionTestCases = [
    {
      id: 'non_streaming_01',
      question: 'Which planet do humans live on?',
      expected: 'earth',
      testId:
        'tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_non_streaming[client_with_models-txt=ollama/llama3.2:3b-instruct-fp16-inference:chat_completion:non_streaming_01]',
    },
    {
      id: 'non_streaming_02',
      question: 'Which planet has rings around it with a name starting with letter S?',
      expected: 'saturn',
      testId:
        'tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_non_streaming[client_with_models-txt=ollama/llama3.2:3b-instruct-fp16-inference:chat_completion:non_streaming_02]',
    },
  ];

  const streamingTestCases = [
    {
      id: 'streaming_01',
      question: "What's the name of the Sun in latin?",
      expected: 'sol',
      testId:
        'tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_streaming[client_with_models-txt=ollama/llama3.2:3b-instruct-fp16-inference:chat_completion:streaming_01]',
    },
    {
      id: 'streaming_02',
      question: 'What is the name of the US captial?',
      expected: 'washington',
      testId:
        'tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_streaming[client_with_models-txt=ollama/llama3.2:3b-instruct-fp16-inference:chat_completion:streaming_02]',
    },
  ];

  test.each(chatCompletionTestCases)(
    'chat completion non-streaming: $id',
    async ({ question, expected, testId }) => {
      const client = createTestClient(testId);
      const textModel = requireTextModel();

      const response = await client.chat.completions.create({
        model: textModel,
        messages: [
          {
            role: 'user',
            content: question,
          },
        ],
        stream: false,
      });

      // Non-streaming responses have choices with message property
      const choice = response.choices[0];
      expect(choice).toBeDefined();
      if (!choice || !('message' in choice)) {
        throw new Error('Expected non-streaming response with message');
      }
      const content = choice.message.content;
      expect(content).toBeDefined();
      const messageContent = typeof content === 'string' ? content.toLowerCase().trim() : '';
      expect(messageContent.length).toBeGreaterThan(0);
      expect(messageContent).toContain(expected.toLowerCase());
    },
  );

  test.each(streamingTestCases)('chat completion streaming: $id', async ({ question, expected, testId }) => {
    const client = createTestClient(testId);
    const textModel = requireTextModel();

    const stream = await client.chat.completions.create({
      model: textModel,
      messages: [{ role: 'user', content: question }],
      stream: true,
    });

    const streamedContent: string[] = [];
    for await (const chunk of stream) {
      if (chunk.choices && chunk.choices.length > 0 && chunk.choices[0]?.delta?.content) {
        streamedContent.push(chunk.choices[0].delta.content);
      }
    }

    expect(streamedContent.length).toBeGreaterThan(0);
    const fullContent = streamedContent.join('').toLowerCase().trim();
    expect(fullContent).toContain(expected.toLowerCase());
  });
});
