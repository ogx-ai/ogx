import { describe, test, expect } from "@jest/globals";

// Extract the exact processChunk function implementation for testing
function createProcessChunk() {
  return (chunk: unknown): { text: string | null; isToolCall: boolean } => {
    const chunkObj = chunk as Record<string, unknown>;

    // Helper function to check if content contains function call JSON
    const containsToolCall = (content: string): boolean => {
      return (
        content.includes('"type": "function"') ||
        content.includes('"name": "file_search"') ||
        content.includes('"parameters":') ||
        !!content.match(/\{"type":\s*"function".*?\}/)
      );
    };

    // Check if this chunk contains a tool call (function call)
    let isToolCall = false;

    // Check direct chunk content if it's a string
    if (typeof chunk === "string") {
      isToolCall = containsToolCall(chunk);
    }

    // Check delta structures
    if (
      chunkObj?.delta &&
      typeof chunkObj.delta === "object" &&
      chunkObj.delta !== null
    ) {
      const delta = chunkObj.delta as Record<string, unknown>;
      if ("tool_calls" in delta) {
        isToolCall = true;
      }
      if (typeof delta.text === "string") {
        if (containsToolCall(delta.text)) {
          isToolCall = true;
        }
      }
    }

    // Check event structures
    if (
      chunkObj?.event &&
      typeof chunkObj.event === "object" &&
      chunkObj.event !== null
    ) {
      const event = chunkObj.event as Record<string, unknown>;

      // Check event payload
      if (
        event?.payload &&
        typeof event.payload === "object" &&
        event.payload !== null
      ) {
        const payload = event.payload as Record<string, unknown>;
        if (typeof payload.content === "string") {
          if (containsToolCall(payload.content)) {
            isToolCall = true;
          }
        }

        // Check payload delta
        if (
          payload?.delta &&
          typeof payload.delta === "object" &&
          payload.delta !== null
        ) {
          const delta = payload.delta as Record<string, unknown>;
          if (typeof delta.text === "string") {
            if (containsToolCall(delta.text)) {
              isToolCall = true;
            }
          }
        }
      }

      // Check event delta
      if (
        event?.delta &&
        typeof event.delta === "object" &&
        event.delta !== null
      ) {
        const delta = event.delta as Record<string, unknown>;
        if (typeof delta.text === "string") {
          if (containsToolCall(delta.text)) {
            isToolCall = true;
          }
        }
        if (typeof delta.content === "string") {
          if (containsToolCall(delta.content)) {
            isToolCall = true;
          }
        }
      }
    }

    // if it's a tool call, skip it (don't display in chat)
    if (isToolCall) {
      return { text: null, isToolCall: true };
    }

    // Extract text content from various chunk formats
    let text: string | null = null;

    // Helper function to extract clean text content, filtering out function calls
    const extractCleanText = (content: string): string | null => {
      if (containsToolCall(content)) {
        try {
          // Try to parse and extract non-function call parts
          const jsonMatch = content.match(
            /\{"type":\s*"function"[^}]*\}[^}]*\}/
          );
          if (jsonMatch) {
            const jsonPart = jsonMatch[0];
            const parsedJson = JSON.parse(jsonPart);

            // If it's a function call, extract text after JSON
            if (parsedJson.type === "function") {
              const textAfterJson = content
                .substring(content.indexOf(jsonPart) + jsonPart.length)
                .trim();
              return textAfterJson || null;
            }
          }
          // If we can't parse it properly, skip the whole thing
          return null;
        } catch {
          return null;
        }
      }
      return content;
    };

    // Try direct delta text
    if (
      chunkObj?.delta &&
      typeof chunkObj.delta === "object" &&
      chunkObj.delta !== null
    ) {
      const delta = chunkObj.delta as Record<string, unknown>;
      if (typeof delta.text === "string") {
        text = extractCleanText(delta.text);
      }
    }

    // Try event structures
    if (
      !text &&
      chunkObj?.event &&
      typeof chunkObj.event === "object" &&
      chunkObj.event !== null
    ) {
      const event = chunkObj.event as Record<string, unknown>;

      // Try event payload content
      if (
        event?.payload &&
        typeof event.payload === "object" &&
        event.payload !== null
      ) {
        const payload = event.payload as Record<string, unknown>;

        // Try direct payload content
        if (typeof payload.content === "string") {
          text = extractCleanText(payload.content);
        }

        // Try turn_complete event structure: payload.turn.output_message.content
        if (
          !text &&
          payload?.turn &&
          typeof payload.turn === "object" &&
          payload.turn !== null
        ) {
          const turn = payload.turn as Record<string, unknown>;
          if (
            turn?.output_message &&
            typeof turn.output_message === "object" &&
            turn.output_message !== null
          ) {
            const outputMessage = turn.output_message as Record<
              string,
              unknown
            >;
            if (typeof outputMessage.content === "string") {
              text = extractCleanText(outputMessage.content);
            }
          }

          // Fallback to model_response in steps if no output_message
          if (
            !text &&
            turn?.steps &&
            Array.isArray(turn.steps) &&
            turn.steps.length > 0
          ) {
            for (const step of turn.steps) {
              if (step && typeof step === "object" && step !== null) {
                const stepObj = step as Record<string, unknown>;
                if (
                  stepObj?.model_response &&
                  typeof stepObj.model_response === "object" &&
                  stepObj.model_response !== null
                ) {
                  const modelResponse = stepObj.model_response as Record<
                    string,
                    unknown
                  >;
                  if (typeof modelResponse.content === "string") {
                    text = extractCleanText(modelResponse.content);
                    break;
                  }
                }
              }
            }
          }
        }

        // Try payload delta
        if (
          !text &&
          payload?.delta &&
          typeof payload.delta === "object" &&
          payload.delta !== null
        ) {
          const delta = payload.delta as Record<string, unknown>;
          if (typeof delta.text === "string") {
            text = extractCleanText(delta.text);
          }
        }
      }

      // Try event delta
      if (
        !text &&
        event?.delta &&
        typeof event.delta === "object" &&
        event.delta !== null
      ) {
        const delta = event.delta as Record<string, unknown>;
        if (typeof delta.text === "string") {
          text = extractCleanText(delta.text);
        }
        if (!text && typeof delta.content === "string") {
          text = extractCleanText(delta.content);
        }
      }
    }

    // Try choices structure (ChatML format)
    if (
      !text &&
      chunkObj?.choices &&
      Array.isArray(chunkObj.choices) &&
      chunkObj.choices.length > 0
    ) {
      const choice = chunkObj.choices[0] as Record<string, unknown>;
      if (
        choice?.delta &&
        typeof choice.delta === "object" &&
        choice.delta !== null
      ) {
        const delta = choice.delta as Record<string, unknown>;
        if (typeof delta.content === "string") {
          text = extractCleanText(delta.content);
        }
      }
    }

    // Try direct string content
    if (!text && typeof chunk === "string") {
      text = extractCleanText(chunk);
    }

    return { text, isToolCall: false };
  };
}

describe("Chunk Processor", () => {
  const processChunk = createProcessChunk();

  describe("Real Event Structures", () => {
    test("handles turn_complete event with cancellation policy response", () => {
      const chunk = {
        event: {
          payload: {
            event_type: "turn_complete",
            turn: {
              turn_id: "50a2d6b7-49ed-4d1e-b1c2-6d68b3f726db",
              session_id: "e7f62b8e-518c-4450-82df-e65fe49f27a3",
              input_messages: [
                {
                  role: "user",
                  content: "nice, what's the cancellation policy?",
                  context: null,
                },
              ],
              steps: [
                {
                  turn_id: "50a2d6b7-49ed-4d1e-b1c2-6d68b3f726db",
                  step_id: "54074310-af42-414c-9ffe-fba5b2ead0ad",
                  started_at: "2025-08-27T18:15:25.870703Z",
                  completed_at: "2025-08-27T18:15:51.288993Z",
                  step_type: "inference",
                  model_response: {
                    role: "assistant",
                    content:
                      "According to the search results, the cancellation policy for Red Hat Summit is as follows:\n\n* Cancellations must be received by 5 PM EDT on April 18, 2025 for a 50% refund of the registration fee.\n* No refunds will be given for cancellations received after 5 PM EDT on April 18, 2025.\n* Cancellation of travel reservations and hotel reservations are the responsibility of the registrant.",
                    stop_reason: "end_of_turn",
                    tool_calls: [],
                  },
                },
              ],
              output_message: {
                role: "assistant",
                content:
                  "According to the search results, the cancellation policy for Red Hat Summit is as follows:\n\n* Cancellations must be received by 5 PM EDT on April 18, 2025 for a 50% refund of the registration fee.\n* No refunds will be given for cancellations received after 5 PM EDT on April 18, 2025.\n* Cancellation of travel reservations and hotel reservations are the responsibility of the registrant.",
                stop_reason: "end_of_turn",
                tool_calls: [],
              },
              output_attachments: [],
              started_at: "2025-08-27T18:15:25.868548Z",
              completed_at: "2025-08-27T18:15:51.289262Z",
            },
          },
        },
      };

      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toContain(
        "According to the search results, the cancellation policy for Red Hat Summit is as follows:"
      );
      expect(result.text).toContain("5 PM EDT on April 18, 2025");
    });

    test("handles turn_complete event with address response", () => {
      const chunk = {
        event: {
          payload: {
            event_type: "turn_complete",
            turn: {
              turn_id: "2f4a1520-8ecc-4cb7-bb7b-886939e042b0",
              session_id: "e7f62b8e-518c-4450-82df-e65fe49f27a3",
              input_messages: [
                {
                  role: "user",
                  content: "what's francisco's address",
                  context: null,
                },
              ],
              steps: [
                {
                  turn_id: "2f4a1520-8ecc-4cb7-bb7b-886939e042b0",
                  step_id: "c13dd277-1acb-4419-8fbf-d5e2f45392ea",
                  started_at: "2025-08-27T18:14:52.558761Z",
                  completed_at: "2025-08-27T18:15:11.306032Z",
                  step_type: "inference",
                  model_response: {
                    role: "assistant",
                    content:
                      "Francisco Arceo's address is:\n\nRed Hat\nUnited States\n17 Primrose Ln \nBasking Ridge New Jersey 07920",
                    stop_reason: "end_of_turn",
                    tool_calls: [],
                  },
                },
              ],
              output_message: {
                role: "assistant",
                content:
                  "Francisco Arceo's address is:\n\nRed Hat\nUnited States\n17 Primrose Ln \nBasking Ridge New Jersey 07920",
                stop_reason: "end_of_turn",
                tool_calls: [],
              },
              output_attachments: [],
              started_at: "2025-08-27T18:14:52.553707Z",
              completed_at: "2025-08-27T18:15:11.306729Z",
            },
          },
        },
      };

      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toContain("Francisco Arceo's address is:");
      expect(result.text).toContain("17 Primrose Ln");
      expect(result.text).toContain("Basking Ridge New Jersey 07920");
    });

    test("handles turn_complete event with ticket cost response", () => {
      const chunk = {
        event: {
          payload: {
            event_type: "turn_complete",
            turn: {
              turn_id: "7ef244a3-efee-42ca-a9c8-942865251002",
              session_id: "e7f62b8e-518c-4450-82df-e65fe49f27a3",
              input_messages: [
                {
                  role: "user",
                  content: "what was the ticket cost for summit?",
                  context: null,
                },
              ],
              steps: [
                {
                  turn_id: "7ef244a3-efee-42ca-a9c8-942865251002",
                  step_id: "7651dda0-315a-472d-b1c1-3c2725f55bc5",
                  started_at: "2025-08-27T18:14:21.710611Z",
                  completed_at: "2025-08-27T18:14:39.706452Z",
                  step_type: "inference",
                  model_response: {
                    role: "assistant",
                    content:
                      "The ticket cost for the Red Hat Summit was $999.00 for a conference pass.",
                    stop_reason: "end_of_turn",
                    tool_calls: [],
                  },
                },
              ],
              output_message: {
                role: "assistant",
                content:
                  "The ticket cost for the Red Hat Summit was $999.00 for a conference pass.",
                stop_reason: "end_of_turn",
                tool_calls: [],
              },
              output_attachments: [],
              started_at: "2025-08-27T18:14:21.705289Z",
              completed_at: "2025-08-27T18:14:39.706752Z",
            },
          },
        },
      };

      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe(
        "The ticket cost for the Red Hat Summit was $999.00 for a conference pass."
      );
    });
  });

  describe("Function Call Detection", () => {
    test("detects function calls in direct string chunks", () => {
      const chunk =
        '{"type": "function", "name": "file_search", "parameters": {"query": "test"}}';
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(true);
      expect(result.text).toBe(null);
    });

    test("detects function calls in event payload content", () => {
      const chunk = {
        event: {
          payload: {
            content:
              '{"type": "function", "name": "file_search", "parameters": {"query": "test"}}',
          },
        },
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(true);
      expect(result.text).toBe(null);
    });

    test("detects tool_calls in delta structure", () => {
      const chunk = {
        delta: {
          tool_calls: [{ function: { name: "file_search" } }],
        },
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(true);
      expect(result.text).toBe(null);
    });

    test("detects function call in mixed content but skips it", () => {
      const chunk =
        '{"type": "function", "name": "file_search", "parameters": {"query": "test"}} Based on the search results, here is your answer.';
      const result = processChunk(chunk);
      // This is detected as a tool call and skipped entirely - the implementation prioritizes safety
      expect(result.isToolCall).toBe(true);
      expect(result.text).toBe(null);
    });
  });

  describe("Text Extraction", () => {
    test("extracts text from direct string chunks", () => {
      const chunk = "Hello, this is a normal response.";
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe("Hello, this is a normal response.");
    });

    test("extracts text from delta structure", () => {
      const chunk = {
        delta: {
          text: "Hello, this is a normal response.",
        },
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe("Hello, this is a normal response.");
    });

    test("extracts text from choices structure", () => {
      const chunk = {
        choices: [
          {
            delta: {
              content: "Hello, this is a normal response.",
            },
          },
        ],
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe("Hello, this is a normal response.");
    });

    test("prioritizes output_message over model_response in turn structure", () => {
      const chunk = {
        event: {
          payload: {
            turn: {
              steps: [
                {
                  model_response: {
                    content: "Model response content.",
                  },
                },
              ],
              output_message: {
                content: "Final output message content.",
              },
            },
          },
        },
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe("Final output message content.");
    });

    test("falls back to model_response when no output_message", () => {
      const chunk = {
        event: {
          payload: {
            turn: {
              steps: [
                {
                  model_response: {
                    content: "This is from the model response.",
                  },
                },
              ],
            },
          },
        },
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe("This is from the model response.");
    });
  });

  describe("Edge Cases", () => {
    test("handles empty chunks", () => {
      const result = processChunk("");
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe("");
    });

    test("handles null chunks", () => {
      const result = processChunk(null);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe(null);
    });

    test("handles undefined chunks", () => {
      const result = processChunk(undefined);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe(null);
    });

    test("handles chunks with no text content", () => {
      const chunk = {
        event: {
          metadata: {
            timestamp: "2024-01-01",
          },
        },
      };
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(false);
      expect(result.text).toBe(null);
    });

    test("handles malformed JSON in function calls gracefully", () => {
      const chunk =
        '{"type": "function", "name": "file_search"} incomplete json';
      const result = processChunk(chunk);
      expect(result.isToolCall).toBe(true);
      expect(result.text).toBe(null);
    });
  });
});
