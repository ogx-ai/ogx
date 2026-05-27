import { describe, test, expect } from "@jest/globals";

/**
 * Tests for processing Responses API streaming chunks.
 *
 * The Responses API uses structured event types like:
 * - response.created: Initial response creation
 * - response.output_text.delta: Text content delta
 * - response.completed: Final completion event
 */

type ChunkResult = {
  text: string | null;
  responseId?: string;
};

function processResponseChunk(chunk: unknown): ChunkResult {
  const chunkObj = chunk as Record<string, unknown>;

  // Handle response.output_text.delta events
  if (chunkObj.type === "response.output_text.delta" && chunkObj.text) {
    return { text: chunkObj.text as string };
  }

  // Handle response.created event
  if (chunkObj.type === "response.created" && chunkObj.response) {
    return {
      text: null,
      responseId: (chunkObj.response as { id: string }).id,
    };
  }

  // Handle response.completed event
  if (chunkObj.type === "response.completed" && chunkObj.response) {
    return {
      text: null,
      responseId: (chunkObj.response as { id: string }).id,
    };
  }

  // Handle response.output_item.added with content
  if (chunkObj.type === "response.output_item.added") {
    const item = chunkObj.item as Record<string, unknown>;
    if (item?.content && typeof item.content === "string") {
      return { text: item.content };
    }
  }

  return { text: null };
}

describe("processResponseChunk", () => {
  describe("text delta events", () => {
    test("extracts text from response.output_text.delta", () => {
      const chunk = {
        type: "response.output_text.delta",
        text: "Hello ",
      };
      expect(processResponseChunk(chunk)).toEqual({ text: "Hello " });
    });

    test("handles empty text delta", () => {
      const chunk = {
        type: "response.output_text.delta",
        text: "",
      };
      // Empty string is falsy, so this returns null
      expect(processResponseChunk(chunk)).toEqual({ text: null });
    });

    test("handles multi-line text delta", () => {
      const chunk = {
        type: "response.output_text.delta",
        text: "Line 1\nLine 2\n",
      };
      expect(processResponseChunk(chunk)).toEqual({
        text: "Line 1\nLine 2\n",
      });
    });
  });

  describe("response lifecycle events", () => {
    test("extracts responseId from response.created", () => {
      const chunk = {
        type: "response.created",
        response: { id: "resp_abc123" },
      };
      expect(processResponseChunk(chunk)).toEqual({
        text: null,
        responseId: "resp_abc123",
      });
    });

    test("extracts responseId from response.completed", () => {
      const chunk = {
        type: "response.completed",
        response: { id: "resp_abc123" },
      };
      expect(processResponseChunk(chunk)).toEqual({
        text: null,
        responseId: "resp_abc123",
      });
    });
  });

  describe("output item events", () => {
    test("extracts content from response.output_item.added", () => {
      const chunk = {
        type: "response.output_item.added",
        item: { content: "Some content" },
      };
      expect(processResponseChunk(chunk)).toEqual({ text: "Some content" });
    });

    test("ignores output_item.added without string content", () => {
      const chunk = {
        type: "response.output_item.added",
        item: { type: "message" },
      };
      expect(processResponseChunk(chunk)).toEqual({ text: null });
    });
  });

  describe("unknown events", () => {
    test("returns null for unknown event types", () => {
      const chunk = {
        type: "response.unknown_event",
        data: "something",
      };
      expect(processResponseChunk(chunk)).toEqual({ text: null });
    });

    test("returns null for empty object", () => {
      expect(processResponseChunk({})).toEqual({ text: null });
    });

    test("returns null for non-object input", () => {
      expect(processResponseChunk("plain string")).toEqual({ text: null });
    });
  });
});
