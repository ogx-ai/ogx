import {
  extractTextFromContentPart,
  extractDisplayableText,
} from "./format-message-content";
import { ChatMessage } from "@/lib/types";

describe("extractTextFromContentPart", () => {
  it("should return an empty string for null or undefined input", () => {
    expect(extractTextFromContentPart(null)).toBe("");
    expect(extractTextFromContentPart(undefined)).toBe("");
  });

  it("should return the string itself if input is a string", () => {
    expect(extractTextFromContentPart("Hello, world!")).toBe("Hello, world!");
    expect(extractTextFromContentPart("")).toBe("");
  });

  it("should extract text from an array of text content objects", () => {
    const content = [{ type: "text", text: "Which planet do humans live on?" }];
    expect(extractTextFromContentPart(content)).toBe(
      "Which planet do humans live on?"
    );
  });

  it("should join text from multiple text content objects in an array", () => {
    const content = [
      { type: "text", text: "Hello," },
      { type: "text", text: "world!" },
    ];
    expect(extractTextFromContentPart(content)).toBe("Hello, world!");
  });

  it("should handle mixed text and image_url types in an array", () => {
    const content = [
      { type: "text", text: "Look at this:" },
      { type: "image_url", image_url: { url: "http://example.com/image.png" } },
      { type: "text", text: "It's an image." },
    ];
    expect(extractTextFromContentPart(content)).toBe(
      "Look at this: [Image] It's an image."
    );
  });

  it("should return '[Image]' for an array with only an image_url object", () => {
    const content = [
      { type: "image_url", image_url: { url: "http://example.com/image.png" } },
    ];
    expect(extractTextFromContentPart(content)).toBe("[Image]");
  });

  it("should return an empty string for an empty array", () => {
    expect(extractTextFromContentPart([])).toBe("");
  });

  it("should handle arrays with plain strings", () => {
    const content = ["This is", " a test."] as unknown;
    expect(extractTextFromContentPart(content)).toBe("This is  a test.");
  });

  it("should filter out malformed or unrecognized objects in an array", () => {
    const content = [
      { type: "text", text: "Valid" },
      { type: "unknown" },
      { text: "Missing type" },
      null,
      undefined,
      { type: "text", noTextProperty: true },
    ] as unknown;
    expect(extractTextFromContentPart(content)).toBe("Valid");
  });

  it("should handle an array of mixed valid items and plain strings", () => {
    const content = [
      { type: "text", text: "First part." },
      "Just a string.",
      { type: "image_url", image_url: { url: "http://example.com/image.png" } },
      { type: "text", text: "Last part." },
    ] as unknown;
    expect(extractTextFromContentPart(content)).toBe(
      "First part. Just a string. [Image] Last part."
    );
  });
});

describe("extractDisplayableText (composite function)", () => {
  const mockFormatToolCallToString = (toolCall: {
    function?: { name?: string; arguments?: unknown };
  }) => {
    if (!toolCall || !toolCall.function || !toolCall.function.name) return "";
    const args = toolCall.function.arguments
      ? JSON.stringify(toolCall.function.arguments)
      : "";
    return `${toolCall.function.name}(${args})`;
  };

  it("should return empty string for null or undefined message", () => {
    expect(extractDisplayableText(null)).toBe("");
    expect(extractDisplayableText(undefined)).toBe("");
  });

  it("should return only content part if no tool calls", () => {
    const message: ChatMessage = {
      role: "assistant",
      content: "Hello there!",
    };
    expect(extractDisplayableText(message)).toBe("Hello there!");
  });

  it("should return only content part for complex content if no tool calls", () => {
    const message: ChatMessage = {
      role: "user",
      content: [
        { type: "text", text: "Part 1" },
        { type: "text", text: "Part 2" },
      ],
    };
    expect(extractDisplayableText(message)).toBe("Part 1 Part 2");
  });

  it("should return only formatted tool call if content is empty or null", () => {
    const toolCall = {
      function: { name: "search", arguments: { query: "cats" } },
    };
    const messageWithEffectivelyEmptyContent: ChatMessage = {
      role: "assistant",
      content: "",
      tool_calls: [toolCall],
    };
    expect(extractDisplayableText(messageWithEffectivelyEmptyContent)).toBe(
      mockFormatToolCallToString(toolCall)
    );

    const messageWithEmptyContent: ChatMessage = {
      role: "assistant",
      content: "",
      tool_calls: [toolCall],
    };
    expect(extractDisplayableText(messageWithEmptyContent)).toBe(
      mockFormatToolCallToString(toolCall)
    );
  });

  it("should combine content and formatted tool call", () => {
    const toolCall = {
      function: { name: "calculator", arguments: { expr: "2+2" } },
    };
    const message: ChatMessage = {
      role: "assistant",
      content: "The result is:",
      tool_calls: [toolCall],
    };
    const expectedToolCallStr = mockFormatToolCallToString(toolCall);
    expect(extractDisplayableText(message)).toBe(
      `The result is: ${expectedToolCallStr}`
    );
  });

  it("should handle message with content an array and a tool call", () => {
    const toolCall = {
      function: { name: "get_weather", arguments: { city: "London" } },
    };
    const message: ChatMessage = {
      role: "assistant",
      content: [
        { type: "text", text: "Okay, checking weather for" },
        { type: "text", text: "London." },
      ],
      tool_calls: [toolCall],
    };
    const expectedToolCallStr = mockFormatToolCallToString(toolCall);
    expect(extractDisplayableText(message)).toBe(
      `Okay, checking weather for London. ${expectedToolCallStr}`
    );
  });

  it("should return only content if tool_calls array is empty or undefined", () => {
    const messageEmptyToolCalls: ChatMessage = {
      role: "assistant",
      content: "No tools here.",
      tool_calls: [],
    };
    expect(extractDisplayableText(messageEmptyToolCalls)).toBe(
      "No tools here."
    );

    const messageUndefinedToolCalls: ChatMessage = {
      role: "assistant",
      content: "Still no tools.",
      tool_calls: undefined,
    };
    expect(extractDisplayableText(messageUndefinedToolCalls)).toBe(
      "Still no tools."
    );
  });
});
